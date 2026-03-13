# binoc-plugin-hdf5

**WARNING:** EXPERIMENTAL, relies on https://github.com/parkan/binoc/tree/native-plugin-registration, intended to demonstrate utility of plumbing through custom data shapes

HDF5 comparator plugin for [binoc](https://github.com/harvard-lil/binoc). Compares HDF5 file structure -- groups, datasets, shapes, dtypes, and attributes -- without reading bulk data.

Includes a rename detector transformer that collapses noisy add/remove pairs into clean rename nodes when the internal structure of two groups matches.

## Build

Requires HDF5 dev headers (`libhdf5-dev` / `hdf5-devel`), or use the `static` feature to build from source (needs cmake):

```
cargo build --features static
```

## Usage

Register the plugin into a binoc `PluginRegistry`:

```rust
let mut registry = binoc_stdlib::default_registry();
binoc_plugin_hdf5::register(&mut registry);
```

Or run the included example to diff two files directly:

```
cargo run --example diff_files -- fixtures/satellite-beams-v1.h5 fixtures/satellite-beams-v2.h5
```

The two fixture files contain identical beam data under different group names (`BEAM0000`/`BEAM0001` vs `BEAM0101`/`BEAM0110`), simulating the kind of group renumbering common in satellite HDF5 products. Without the rename detector, the raw diff produces 12 nodes -- 6 adds and 6 removes:

```json
{
  "kind": "modify",
  "item_type": "hdf5_file",
  "children": [
    { "kind": "add", "item_type": "hdf5_group", "path": ".../BEAM0101", "summary": "Group added (2 datasets, 0 attrs)" },
    { "kind": "add", "item_type": "hdf5_dataset", "path": ".../BEAM0101/elevation" },
    { "kind": "add", "item_type": "hdf5_dataset", "path": ".../BEAM0101/latitude" },
    { "kind": "add", "item_type": "hdf5_group", "path": ".../BEAM0110", "summary": "Group added (2 datasets, 0 attrs)" },
    { "kind": "add", "item_type": "hdf5_dataset", "path": ".../BEAM0110/elevation" },
    { "kind": "add", "item_type": "hdf5_dataset", "path": ".../BEAM0110/latitude" },
    { "kind": "remove", "item_type": "hdf5_group", "path": ".../BEAM0000", "summary": "Group removed (2 datasets, 0 attrs)" },
    { "kind": "remove", "item_type": "hdf5_dataset", "path": ".../BEAM0000/elevation" },
    { "kind": "remove", "item_type": "hdf5_dataset", "path": ".../BEAM0000/latitude" },
    { "kind": "remove", "item_type": "hdf5_group", "path": ".../BEAM0001", "summary": "Group removed (2 datasets, 0 attrs)" },
    { "kind": "remove", "item_type": "hdf5_dataset", "path": ".../BEAM0001/elevation" },
    { "kind": "remove", "item_type": "hdf5_dataset", "path": ".../BEAM0001/latitude" }
  ]
}
```

With the rename detector (`--detect-renames`), those 12 nodes collapse to 2:

```
cargo run --example diff_files -- fixtures/satellite-beams-v1.h5 fixtures/satellite-beams-v2.h5 --detect-renames
```

```json
{
  "kind": "modify",
  "item_type": "hdf5_file",
  "children": [
    {
      "kind": "rename",
      "item_type": "hdf5_group",
      "path": ".../BEAM0101",
      "source_path": ".../BEAM0000",
      "summary": "BEAM0000 → BEAM0101 (structure unchanged)",
      "tags": ["binoc-hdf5.group-rename"],
      "details": { "datasets": ["elevation", "latitude"] }
    },
    {
      "kind": "rename",
      "item_type": "hdf5_group",
      "path": ".../BEAM0110",
      "source_path": ".../BEAM0001",
      "summary": "BEAM0001 → BEAM0110 (structure unchanged)",
      "tags": ["binoc-hdf5.group-rename"],
      "details": { "datasets": ["elevation", "latitude"] }
    }
  ]
}
```

The rename detector works by fingerprinting each group's direct child datasets (name, shape, dtype) and matching removed/added groups with identical fingerprints. It requires the `CustomReopenedData` mechanism in binoc-core to transport typed `Hdf5DataPair` from the comparator to the transformer.

## Fixtures

- `satellite-beams-v1.h5` / `satellite-beams-v2.h5` -- beam rename detection demo
- `example-femm-3d.h5` / `example-femm-3d-modified.h5` -- dataset removal demo, sourced from [openPMD-example-datasets](https://github.com/openPMD/openPMD-example-datasets)
