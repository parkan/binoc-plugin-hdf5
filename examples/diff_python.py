"""Diff two HDF5 files from Python using the native Rust plugin.

Demonstrates the full pipeline: the comparator caches typed Hdf5DataPair
via CustomReopenedData, and the rename detector transformer consumes it
to collapse add/remove pairs into rename nodes -- all running natively
in Rust, triggered from Python.

Usage:
    python examples/diff_python.py fixtures/satellite-beams-v1.h5 fixtures/satellite-beams-v2.h5
"""
import sys
import json
import binoc


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <left.h5> <right.h5>", file=sys.stderr)
        sys.exit(1)

    left, right = sys.argv[1], sys.argv[2]

    # build registry with entry-point-discovered plugins (including ours)
    registry = binoc.PluginRegistry.default()
    binoc.discover_plugins(registry)

    # configure to use the HDF5 comparator and rename detector
    config = binoc.Config(
        comparators=["binoc-hdf5.hdf5"],
        transformers=["binoc-hdf5.rename_detector"],
    )

    # diff -- the full pipeline runs in Rust:
    # comparator caches Hdf5DataPair -> transformer downcasts and rewrites
    migration = binoc.diff(left, right, config=config, registry=registry)
    print(json.dumps(json.loads(migration.to_json()), indent=2))


if __name__ == "__main__":
    main()
