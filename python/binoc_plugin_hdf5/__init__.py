"""binoc-plugin-hdf5: native HDF5 comparator and rename detector for binoc."""

from binoc_plugin_hdf5._binoc_plugin_hdf5 import _register_native


def register(registry):
    """Register native Rust HDF5 comparator and rename detector.

    Called automatically by binoc's entry-point discovery, or manually:

        registry = binoc.PluginRegistry.default()
        binoc_plugin_hdf5.register(registry)
    """
    _register_native(registry._inner_ptr())
