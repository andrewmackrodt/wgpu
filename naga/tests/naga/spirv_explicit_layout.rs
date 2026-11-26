/*!
SPIR-V backend: a type used both in buffer memory and in `Function`, `Private`
or `Workgroup` memory.

Explicit layout decorations (`Offset`, `ArrayStride`, `MatrixStride`) are
written only on types reachable from buffer variables, but a struct or array
that is also copied into a function-local variable (`var s = buf[i];`) is one
type in the IR and was written as one SPIR-V type, so the `Function` variable
pointed to a type with explicit layout, which Vulkan forbids
(`VUID-StandaloneSpirv-None-10684`).

These tests read the written SPIR-V for:

- pointers in the `Workgroup` storage class, and from SPIR-V 1.4 in the
  `Function` and `Private` storage classes, whose pointee carries explicit
  layout decorations anywhere inside it;
- loads, stores and variable initializers whose value type is not the
  pointee type of the pointer (a layout-free copy of a type must be converted
  when its values move to or from buffer memory).
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

use hashbrown::{HashMap, HashSet};
use naga::back::spv;
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
use naga::valid;
use rspirv::dr::Operand;
use spirv::{Decoration, Op, StorageClass};

fn write(source: &str, lang_version: (u8, u8), policy: BoundsCheckPolicy) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("WGSL parse:\n{}", e.emit_to_string(source)));
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("validation failed");
    let options = spv::Options {
        lang_version,
        bounds_check_policies: BoundsCheckPolicies {
            index: policy,
            buffer: policy,
            image_load: policy,
            binding_array: BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        ..spv::Options::default()
    };
    spv::write_vec(&module, &info, &options, None).expect("SPIR-V writer failed")
}

fn id(operand: &Operand) -> u32 {
    match *operand {
        Operand::IdRef(id) => id,
        _ => panic!("not an id: {operand:?}"),
    }
}

/// Every problem found in `words`, as text.
fn problems(words: &[u32], lang_version: (u8, u8)) -> Vec<String> {
    let module = rspirv::dr::load_words(words).expect("rspirv could not load the SPIR-V");

    // Types with explicit layout decorations.
    let mut laid_out = HashSet::new();
    for inst in &module.annotations {
        let decoration = match inst.class.opcode {
            Op::Decorate => &inst.operands[1],
            Op::MemberDecorate => &inst.operands[2],
            _ => continue,
        };
        if let Operand::Decoration(
            Decoration::Offset
            | Decoration::ArrayStride
            | Decoration::MatrixStride
            | Decoration::RowMajor
            | Decoration::ColMajor,
        ) = *decoration
        {
            laid_out.insert(id(&inst.operands[0]));
        }
    }

    // The type's components, and each pointer type's storage class and pointee.
    let mut components: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut pointers = HashMap::new();
    let mut type_of = HashMap::new();
    for inst in &module.types_global_values {
        let Some(result) = inst.result_id else {
            continue;
        };
        match inst.class.opcode {
            Op::TypeStruct => {
                components.insert(result, inst.operands.iter().map(id).collect());
            }
            Op::TypeArray | Op::TypeRuntimeArray => {
                components.insert(result, vec![id(&inst.operands[0])]);
            }
            Op::TypePointer => {
                let Operand::StorageClass(class) = inst.operands[0] else {
                    unreachable!()
                };
                pointers.insert(result, (class, id(&inst.operands[1])));
            }
            _ => {}
        }
        if let Some(ty) = inst.result_type {
            type_of.insert(result, ty);
        }
    }

    let mut problems = vec![];

    // No explicit layout under a `Workgroup` pointer, nor, from SPIR-V 1.4,
    // under a `Function` or `Private` one (as `spirv-val` checks it).
    for (&pointer, &(class, pointee)) in &pointers {
        let forbidden = match class {
            StorageClass::Workgroup => true,
            StorageClass::Function | StorageClass::Private => lang_version >= (1, 4),
            _ => false,
        };
        if !forbidden {
            continue;
        }
        let mut stack = vec![pointee];
        while let Some(ty) = stack.pop() {
            if laid_out.contains(&ty) {
                problems.push(format!(
                    "pointer %{pointer} ({class:?}) reaches %{ty}, which has explicit layout"
                ));
            }
            if let Some(c) = components.get(&ty) {
                stack.extend(c);
            }
        }
    }

    // Loads, stores and initializers match their pointer's pointee type.
    for function in &module.functions {
        for param in &function.parameters {
            type_of.insert(param.result_id.unwrap(), param.result_type.unwrap());
        }
        for block in &function.blocks {
            for inst in &block.instructions {
                if let (Some(result), Some(ty)) = (inst.result_id, inst.result_type) {
                    type_of.insert(result, ty);
                }
            }
        }
    }
    let pointee = |pointer: u32| pointers[&type_of[&pointer]].1;
    let variables = module.types_global_values.iter().chain(
        module
            .functions
            .iter()
            .flat_map(|f| f.blocks.iter().flat_map(|b| b.instructions.iter())),
    );
    for inst in variables {
        match inst.class.opcode {
            Op::Load => {
                let (ty, from) = (inst.result_type.unwrap(), pointee(id(&inst.operands[0])));
                if ty != from {
                    problems.push(format!(
                        "OpLoad %{} of type %{ty} from a pointer to %{from}",
                        inst.result_id.unwrap()
                    ));
                }
            }
            Op::Store => {
                let (to, value) = (pointee(id(&inst.operands[0])), id(&inst.operands[1]));
                if type_of[&value] != to {
                    problems.push(format!(
                        "OpStore of %{value} (type %{}) to a pointer to %{to}",
                        type_of[&value]
                    ));
                }
            }
            Op::Variable if inst.operands.len() > 1 => {
                let (to, init) = (
                    pointers[&inst.result_type.unwrap()].1,
                    id(&inst.operands[1]),
                );
                if type_of[&init] != to {
                    problems.push(format!(
                        "OpVariable %{} of %{to} initialized with %{init} (type %{})",
                        inst.result_id.unwrap(),
                        type_of[&init]
                    ));
                }
            }
            _ => {}
        }
    }
    problems
}

/// Check `source` at SPIR-V 1.0 and 1.3 (copies converted member by member)
/// and 1.6 (`OpCopyLogical`), under each bounds check policy.
fn check(source: &str) {
    for lang_version in [(1, 0), (1, 3), (1, 6)] {
        for policy in [
            BoundsCheckPolicy::Unchecked,
            BoundsCheckPolicy::Restrict,
            BoundsCheckPolicy::ReadZeroSkipWrite,
        ] {
            let found = problems(&write(source, lang_version, policy), lang_version);
            assert!(
                found.is_empty(),
                "SPIR-V {lang_version:?}, {policy:?}:\n{}\n\n{source}",
                found.join("\n")
            );
        }
    }
}

/// A struct copied from a storage array into a local, changed and written back.
#[test]
fn storage_element_copied_to_local() {
    check(
        "struct Slot { at: vec4<f32>, light: vec4<f32>, info: vec4<f32> }
         @group(0) @binding(0) var<storage, read_write> slots: array<Slot>;
         fn spawn(k: u32) -> Slot {
             return Slot(vec4(f32(k)), vec4(1.0), vec4(0.0));
         }
         @compute @workgroup_size(64)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
             var s = slots[id.x];
             if (s.info.z != 1.0) {
                 s = spawn(id.x);
                 slots[id.x] = s;
             }
             slots[id.x + 1u].at = s.at;
         }",
    );
}

/// Nested structs, arrays with a stride and matrices in a uniform and a
/// storage buffer, copied whole and in part into locals, and through a
/// function's pointer parameter.
#[test]
fn nested_types_copied_to_locals() {
    check(
        "struct Inner { m: mat3x3<f32>, v: array<vec4<f32>, 2> }
         struct Outer { a: f32, inner: Inner, arr: array<Inner, 3> }
         @group(0) @binding(0) var<storage, read_write> o: Outer;
         @group(0) @binding(1) var<uniform> u: Outer;
         fn bump(p: ptr<function, Inner>) {
             (*p).v[1].x += 1.0;
         }
         @compute @workgroup_size(1)
         fn main(@builtin(local_invocation_index) i: u32) {
             var whole = o;
             var inner = u.inner;
             var arr = u.arr;
             var v = o.inner.v;
             bump(&inner);
             arr[i] = inner;
             whole.inner = arr[i % 3u];
             o = whole;
             o.arr[i] = whole.arr[0];
             o.inner.v = v;
             o.a = whole.inner.m[0].x;
         }",
    );
}

/// A uniform struct whose type is written in a std140-compatible form (a
/// two-row matrix), copied into a local.
#[test]
fn std140_uniform_copied_to_local() {
    check(
        "struct S { m: mat2x2<f32>, arr: array<vec4<f32>, 2> }
         @group(0) @binding(0) var<uniform> u: S;
         @group(0) @binding(1) var<storage, read_write> out: S;
         @compute @workgroup_size(1)
         fn main() {
             var s = u;
             s.m[0].x += 1.0;
             out = s;
         }",
    );
}

/// `Private` (with an initializer) and `Workgroup` (zero-initialized)
/// variables of a buffer's struct type.
#[test]
fn private_and_workgroup_of_buffer_types() {
    check(
        "struct S { a: vec4<f32>, b: array<f32, 4> }
         @group(0) @binding(0) var<storage, read_write> buf: array<S, 4>;
         var<private> p: S = S(vec4(1.0), array(1.0, 2.0, 3.0, 4.0));
         var<private> q: array<S, 2>;
         var<workgroup> w: S;
         @compute @workgroup_size(4)
         fn main(@builtin(local_invocation_index) i: u32) {
             if (i == 0u) {
                 w = buf[0];
             }
             workgroupBarrier();
             q[i % 2u] = p;
             buf[i] = workgroupUniformLoad(&w);
             p = buf[i];
             buf[i].b = q[0].b;
         }",
    );
}

/// A buffer's array loaded as a value and indexed dynamically, which the
/// writer spills to a function-local variable.
#[test]
fn spilled_buffer_array() {
    check(
        "struct S { a: vec4<f32>, b: f32 }
         @group(0) @binding(0) var<storage, read_write> buf: array<array<S, 4>, 2>;
         @compute @workgroup_size(1)
         fn main(@builtin(local_invocation_index) i: u32) {
             let a = buf[0];
             let s = a[i];
             buf[1][i] = s;
         }",
    );
}
