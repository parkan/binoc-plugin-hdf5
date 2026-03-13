# binoc-plugin-hdf5

HDF5 comparator plugin for [binoc](https://github.com/harvard-lil/binoc). Compares HDF5 file structure -- groups, datasets, shapes, dtypes, and attributes -- without reading bulk data.

## Build

Requires HDF5 dev headers (`libhdf5-dev` / `hdf5-devel`), or use the `static` feature to build from source (needs cmake):

```
cargo build --features static
```

On immutable systems, build in a container (see Dockerfile).

## Usage

Register the plugin into a binoc `PluginRegistry`:

```rust
let mut registry = binoc_stdlib::default_registry();
binoc_plugin_hdf5::register(&mut registry);
```

Or run the included example to diff two files directly:

```
cargo run --example diff_files -- fixtures/example-femm-3d.h5 fixtures/example-femm-3d-modified.h5
```

The modified fixture has `data/1/meshes/B/y` deleted. Output:

```json
{
  "kind": "modify",
  "item_type": "hdf5_file",
  "children": [
    {
      "kind": "remove",
      "item_type": "hdf5_dataset",
      "path": "example-femm-3d-modified.h5/data/1/meshes/B/y",
      "summary": "Dataset removed (47, 47, 47) float64",
      "tags": ["binoc-hdf5.dataset-removal", "binoc.schema-change"],
      "details": {
        "dtype": "float64",
        "shape": [47, 47, 47],
        "num_attrs": 2
      }
    }
  ]
}
```

Fixtures sourced from [openPMD-example-datasets](https://github.com/openPMD/openPMD-example-datasets).
