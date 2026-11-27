//! A per-thread tap on the SPIR-V that pipeline creation hands to the driver.
//!
//! When a pipeline stage's module is naga IR, `compile_stage` writes SPIR-V for it (applying the pipeline layout's
//! binding map, the override constants and the device's writer options) and passes those words to
//! `vkCreateShaderModule`. An application that keeps its own store of compiled shaders can capture exactly those words
//! with [`with_spirv_tap`] and later create the same stages from them as passthrough SPIR-V modules, skipping naga.
//!
//! The tap is installed for the calling thread only and only for the duration of the closure, so it sees the stages of
//! pipelines that are created synchronously on that thread. It is not called for stages whose module was already raw
//! SPIR-V.

use core::cell::Cell;

/// The callback: the stage, its entry point and the SPIR-V words handed to `vkCreateShaderModule`.
pub type SpirvTap<'a> = dyn FnMut(naga::ShaderStage, &str, &[u32]) + 'a;

std::thread_local! {
    static TAP: Cell<Option<*mut SpirvTap<'static>>> = const { Cell::new(None) };
}

/// Restores the tap that was installed before, also when the closure panics.
struct Restore(Option<*mut SpirvTap<'static>>);

impl Drop for Restore {
    fn drop(&mut self) {
        TAP.with(|t| t.set(self.0));
    }
}

/// Runs `f` with `tap` called for every stage SPIR-V written by pipeline creation on this thread.
///
/// The tap must not create pipelines or shader modules itself. Taps nest: the innermost one is called, and the outer
/// one is restored when `f` returns or unwinds.
pub fn with_spirv_tap<R>(tap: &mut SpirvTap<'_>, f: impl FnOnce() -> R) -> R {
    // SAFETY: only the lifetime is erased. The pointer is removed from the thread-local before `tap`'s borrow ends
    // (`Restore` runs when this function returns or unwinds), and it is only dereferenced on this thread while it is
    // installed.
    let tap: *mut SpirvTap<'_> = tap;
    let erased: *mut SpirvTap<'static> = unsafe { core::mem::transmute(tap) };
    let _restore = Restore(TAP.with(|t| t.replace(Some(erased))));
    f()
}

/// Hands `words` to this thread's tap, if one is installed.
pub(super) fn tap(stage: naga::ShaderStage, entry_point: &str, words: &[u32]) {
    let Some(tap) = TAP.with(|t| t.take()) else {
        return;
    };
    // The tap is taken out while it runs, so it is never borrowed twice.
    let _restore = Restore(Some(tap));
    // SAFETY: installed by `with_spirv_tap`, whose borrow outlives its installation on this thread.
    unsafe { (*tap)(stage, entry_point, words) };
}
