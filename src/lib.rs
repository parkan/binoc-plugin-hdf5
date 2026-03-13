mod hdf5_compare;

use binoc_core::config::PluginRegistry;
use std::sync::Arc;

pub use hdf5_compare::Hdf5Comparator;

pub fn register(registry: &mut PluginRegistry) {
    registry.register_comparator("binoc-hdf5.hdf5", Arc::new(Hdf5Comparator));
}
