pub mod hdf5_compare;
mod rename_detector;

use binoc_core::config::PluginRegistry;
use std::sync::Arc;

pub use hdf5_compare::Hdf5Comparator;
pub use rename_detector::Hdf5RenameDetector;

pub fn register(registry: &mut PluginRegistry) {
    registry.register_comparator("binoc-hdf5.hdf5", Arc::new(Hdf5Comparator));
    registry.register_transformer("binoc-hdf5.rename_detector", Arc::new(Hdf5RenameDetector));
}

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    /// Register native Rust comparator and transformer into a binoc
    /// PluginRegistry via its raw pointer (obtained from registry._inner_ptr()).
    #[pyfunction]
    fn _register_native(registry_ptr: usize) {
        let registry =
            unsafe { &mut *(registry_ptr as *mut binoc_core::config::PluginRegistry) };
        super::register(registry);
    }

    #[pymodule]
    fn _binoc_plugin_hdf5(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(_register_native, m)?)?;
        Ok(())
    }
}
