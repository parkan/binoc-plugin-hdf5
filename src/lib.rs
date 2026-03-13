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
