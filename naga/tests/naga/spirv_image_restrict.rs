/*!
SPIR-V backend: the `Restrict` image-load bounds check policy must clamp
coordinates with constants of the coordinates' own type.

`write_coordinate_one` built its vector of ones from the signed scalar `1i`
whatever the coordinate type, so a `textureLoad` with `u32` coordinates wrote
`OpConstantComposite %v2uint %int_1 %int_1`: a constituent whose type is not
the vector's component type, which `spirv-val` rejects.
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

use naga::back::spv;
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
use naga::valid;
use rspirv::dr::Operand;
use spirv::Op;

/// SPIR-V words for `source` with every bounds check policy set to `Restrict`.
fn restricted_spirv(source: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("WGSL parse:\n{}", e.emit_to_string(source)));
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("validation failed");
    let options = spv::Options {
        lang_version: (1, 6),
        bounds_check_policies: BoundsCheckPolicies {
            index: BoundsCheckPolicy::Restrict,
            buffer: BoundsCheckPolicy::Restrict,
            image_load: BoundsCheckPolicy::Restrict,
            binding_array: BoundsCheckPolicy::Unchecked,
        },
        ..spv::Options::default()
    };
    spv::write_vec(&module, &info, &options, None).expect("SPIR-V writer failed")
}

/// Every `OpConstantComposite` of vector type whose constituents are not all of
/// the vector's component type, as `(composite id, component type, constituent
/// type)`.
fn mistyped_vector_constants(words: &[u32]) -> Vec<(u32, u32, u32)> {
    let module = rspirv::dr::load_words(words).expect("rspirv could not load the SPIR-V");
    let mut type_of = hashbrown::HashMap::new();
    let mut vector_component = hashbrown::HashMap::new();
    for inst in &module.types_global_values {
        if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
            type_of.insert(id, ty);
        }
        if inst.class.opcode == Op::TypeVector {
            if let Operand::IdRef(component) = inst.operands[0] {
                vector_component.insert(inst.result_id.unwrap(), component);
            }
        }
    }
    let mut bad = vec![];
    for inst in &module.types_global_values {
        if inst.class.opcode != Op::ConstantComposite {
            continue;
        }
        let ty = inst.result_type.unwrap();
        let Some(&component) = vector_component.get(&ty) else {
            continue;
        };
        for operand in &inst.operands {
            if let Operand::IdRef(c) = *operand {
                let c_ty = type_of[&c];
                if c_ty != component {
                    bad.push((inst.result_id.unwrap(), component, c_ty));
                }
            }
        }
    }
    bad
}

fn check(source: &str) {
    let bad = mistyped_vector_constants(&restricted_spirv(source));
    assert!(
        bad.is_empty(),
        "vector constants with constituents of another type (composite, component type, constituent type): {bad:?}\n\n{source}"
    );
}

#[test]
fn restrict_u32_coordinates_2d() {
    check(
        "@group(0) @binding(0) var t: texture_2d<f32>;
         @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
             return textureLoad(t, vec2<u32>(p.xy), 0u);
         }",
    );
}

#[test]
fn restrict_u32_coordinates_3d() {
    check(
        "@group(0) @binding(0) var t: texture_3d<f32>;
         @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
             return textureLoad(t, vec3<u32>(p.xyz), 0);
         }",
    );
}

#[test]
fn restrict_u32_coordinates_arrayed() {
    // The array layer joins the coordinates as a third component of their type
    // (an i32 layer is bitcast to u32 here; a u32 layer beside i32 coordinates
    // is bitcast to i32).
    check(
        "@group(0) @binding(0) var t: texture_2d_array<f32>;
         @group(0) @binding(1) var d: texture_depth_2d_array;
         @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
             let a = textureLoad(t, vec2<u32>(p.xy), i32(p.z), 0u);
             let b = textureLoad(t, vec2<i32>(p.xy), u32(p.z), 0);
             let c = textureLoad(d, vec2<u32>(p.xy), 1u, 0u);
             return a + b + vec4(c);
         }",
    );
}

#[test]
fn restrict_u32_coordinates_storage_multisampled_depth() {
    check(
        "@group(0) @binding(0) var s: texture_storage_2d<rgba8unorm, read>;
         @group(0) @binding(1) var m: texture_multisampled_2d<f32>;
         @group(0) @binding(2) var d: texture_depth_2d;
         @group(0) @binding(3) var dm: texture_depth_multisampled_2d;
         @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
             let c = vec2<u32>(p.xy);
             return textureLoad(s, c) + textureLoad(m, c, 1u)
                 + vec4(textureLoad(d, c, 0u) + textureLoad(dm, c, 0u));
         }",
    );
}

#[test]
fn restrict_i32_coordinates_stay_signed() {
    check(
        "@group(0) @binding(0) var t: texture_2d<f32>;
         @group(0) @binding(1) var a: texture_2d_array<f32>;
         @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
             return textureLoad(t, vec2<i32>(p.xy), 0) + textureLoad(a, vec2<i32>(p.xy), 0, 0);
         }",
    );
}
