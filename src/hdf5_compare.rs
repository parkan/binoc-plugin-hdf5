use std::collections::BTreeMap;
use std::path::Path;

use binoc_core::ir::DiffNode;
use binoc_core::traits::*;
use binoc_core::types::*;

pub struct Hdf5Comparator;

const HDF5_EXTENSIONS: &[&str] = &[".h5", ".he5", ".hdf5", ".hdf", ".nc", ".nc4"];
const HDF5_MEDIA_TYPES: &[&str] = &["application/x-hdf5", "application/x-hdf"];

#[derive(Debug, Clone)]
struct DatasetMeta {
    shape: Vec<usize>,
    dtype: String,
    num_attrs: usize,
}

#[derive(Debug, Clone)]
struct GroupMeta {
    num_attrs: usize,
    num_datasets: usize,
}

#[derive(Debug, Clone)]
enum EntryMeta {
    Group(GroupMeta),
    Dataset(DatasetMeta),
}

fn open_h5(path: &Path) -> BinocResult<hdf5::File> {
    hdf5::File::open(path).map_err(|e| BinocError::Other(format!("hdf5: {e}")))
}

fn read_inventory(path: &Path) -> BinocResult<BTreeMap<String, EntryMeta>> {
    let file = open_h5(path)?;
    let mut entries = BTreeMap::new();
    walk_group(&file, &mut entries)?;
    Ok(entries)
}

fn walk_group(
    group: &hdf5::Group,
    entries: &mut BTreeMap<String, EntryMeta>,
) -> BinocResult<()> {
    let name = group.name();
    let datasets = group
        .datasets()
        .map_err(|e| BinocError::Other(format!("hdf5: {e}")))?;
    let child_groups = group
        .groups()
        .map_err(|e| BinocError::Other(format!("hdf5: {e}")))?;

    let num_attrs = group.attr_names().map(|v| v.len()).unwrap_or(0);

    // skip root group -- it maps to the file-level node
    if name != "/" {
        entries.insert(
            name,
            EntryMeta::Group(GroupMeta {
                num_attrs,
                num_datasets: datasets.len(),
            }),
        );
    }

    for ds in &datasets {
        let shape = ds.shape();
        let dtype = ds
            .dtype()
            .map(|dt| format!("{dt}"))
            .unwrap_or_else(|_| "unknown".into());
        let ds_attrs = ds.attr_names().map(|v| v.len()).unwrap_or(0);
        entries.insert(
            ds.name(),
            EntryMeta::Dataset(DatasetMeta {
                shape,
                dtype,
                num_attrs: ds_attrs,
            }),
        );
    }

    for child in &child_groups {
        walk_group(child, entries)?;
    }

    Ok(())
}

fn fmt_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        "scalar".into()
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", dims.join(", "))
    }
}

// strip leading / from HDF5 internal paths
fn h5_path_to_logical(file_logical: &str, h5_path: &str) -> String {
    let trimmed = h5_path.trim_start_matches('/');
    if trimmed.is_empty() {
        file_logical.to_string()
    } else {
        format!("{file_logical}/{trimmed}")
    }
}

impl Comparator for Hdf5Comparator {
    fn name(&self) -> &str {
        "binoc-hdf5.hdf5"
    }

    fn handles_extensions(&self) -> &[&str] {
        HDF5_EXTENSIONS
    }

    fn handles_media_types(&self) -> &[&str] {
        HDF5_MEDIA_TYPES
    }

    fn compare(&self, pair: &ItemPair, _ctx: &CompareContext) -> BinocResult<CompareResult> {
        match (&pair.left, &pair.right) {
            (Some(left), Some(right)) => self.compare_both(left, right, pair.logical_path()),
            (None, Some(right)) => {
                let inv = read_inventory(&right.physical_path)?;
                let n_groups = inv
                    .values()
                    .filter(|e| matches!(e, EntryMeta::Group(_)))
                    .count();
                let n_datasets = inv
                    .values()
                    .filter(|e| matches!(e, EntryMeta::Dataset(_)))
                    .count();

                let node = DiffNode::new("add", "hdf5_file", &right.logical_path)
                    .with_summary(format!(
                        "New HDF5 file ({} group{}, {} dataset{})",
                        n_groups,
                        if n_groups == 1 { "" } else { "s" },
                        n_datasets,
                        if n_datasets == 1 { "" } else { "s" },
                    ))
                    .with_tag("binoc-hdf5.content-changed")
                    .with_detail("groups", serde_json::json!(n_groups))
                    .with_detail("datasets", serde_json::json!(n_datasets));

                Ok(CompareResult::Leaf(node))
            }
            (Some(left), None) => {
                let inv = read_inventory(&left.physical_path)?;
                let n_groups = inv
                    .values()
                    .filter(|e| matches!(e, EntryMeta::Group(_)))
                    .count();
                let n_datasets = inv
                    .values()
                    .filter(|e| matches!(e, EntryMeta::Dataset(_)))
                    .count();

                let node = DiffNode::new("remove", "hdf5_file", &left.logical_path)
                    .with_summary(format!(
                        "HDF5 file removed ({} group{}, {} dataset{})",
                        n_groups,
                        if n_groups == 1 { "" } else { "s" },
                        n_datasets,
                        if n_datasets == 1 { "" } else { "s" },
                    ))
                    .with_tag("binoc-hdf5.content-changed")
                    .with_detail("groups", serde_json::json!(n_groups))
                    .with_detail("datasets", serde_json::json!(n_datasets));

                Ok(CompareResult::Leaf(node))
            }
            (None, None) => Ok(CompareResult::Identical),
        }
    }
}

impl Hdf5Comparator {
    fn compare_both(
        &self,
        left: &Item,
        right: &Item,
        logical_path: &str,
    ) -> BinocResult<CompareResult> {
        let inv_l = read_inventory(&left.physical_path)?;
        let inv_r = read_inventory(&right.physical_path)?;

        let mut children = Vec::new();

        // entries in both -- check for metadata changes
        for (h5_path, meta_l) in &inv_l {
            if let Some(meta_r) = inv_r.get(h5_path) {
                if let Some(node) = diff_entry(logical_path, h5_path, meta_l, meta_r) {
                    children.push(node);
                }
            }
        }

        // entries added
        for (h5_path, meta) in &inv_r {
            if !inv_l.contains_key(h5_path) {
                children.push(added_entry(logical_path, h5_path, meta));
            }
        }

        // entries removed
        for (h5_path, meta) in &inv_l {
            if !inv_r.contains_key(h5_path) {
                children.push(removed_entry(logical_path, h5_path, meta));
            }
        }

        if children.is_empty() {
            return Ok(CompareResult::Identical);
        }

        let node = DiffNode::new("modify", "hdf5_file", logical_path)
            .with_children(children)
            .with_detail(
                "entries_left",
                serde_json::json!(inv_l.keys().collect::<Vec<_>>()),
            )
            .with_detail(
                "entries_right",
                serde_json::json!(inv_r.keys().collect::<Vec<_>>()),
            );

        Ok(CompareResult::Leaf(node))
    }
}

fn diff_entry(
    file_logical: &str,
    h5_path: &str,
    left: &EntryMeta,
    right: &EntryMeta,
) -> Option<DiffNode> {
    let path = h5_path_to_logical(file_logical, h5_path);

    match (left, right) {
        (EntryMeta::Dataset(dl), EntryMeta::Dataset(dr)) => {
            diff_dataset(&path, h5_path, dl, dr)
        }
        (EntryMeta::Group(gl), EntryMeta::Group(gr)) => diff_group(&path, h5_path, gl, gr),
        // type changed (group became dataset or vice versa) -- emit remove + add
        // handled by the added/removed logic since keys won't match types
        _ => {
            let mut node = DiffNode::new("modify", "hdf5_entry", &path)
                .with_tag("binoc-hdf5.type-change")
                .with_summary("Entry type changed");
            node.tags.insert("binoc-hdf5.schema-change".into());
            node.tags.insert("binoc.schema-change".into());
            Some(node)
        }
    }
}

fn diff_dataset(
    logical_path: &str,
    _h5_path: &str,
    left: &DatasetMeta,
    right: &DatasetMeta,
) -> Option<DiffNode> {
    let shape_changed = left.shape != right.shape;
    let dtype_changed = left.dtype != right.dtype;
    let attrs_changed = left.num_attrs != right.num_attrs;

    if !shape_changed && !dtype_changed && !attrs_changed {
        return None;
    }

    let mut node = DiffNode::new("modify", "hdf5_dataset", logical_path)
        .with_detail("shape_left", serde_json::json!(left.shape))
        .with_detail("shape_right", serde_json::json!(right.shape))
        .with_detail("dtype_left", serde_json::json!(&left.dtype))
        .with_detail("dtype_right", serde_json::json!(&right.dtype))
        .with_detail("attrs_left", serde_json::json!(left.num_attrs))
        .with_detail("attrs_right", serde_json::json!(right.num_attrs));

    if shape_changed {
        node.tags.insert("binoc-hdf5.shape-change".into());
        node.tags.insert("binoc.schema-change".into());
    }
    if dtype_changed {
        node.tags.insert("binoc-hdf5.dtype-change".into());
        node.tags.insert("binoc.schema-change".into());
    }
    if attrs_changed {
        node.tags.insert("binoc-hdf5.attr-change".into());
    }

    let mut parts = Vec::new();
    if shape_changed {
        parts.push(format!(
            "shape {} \u{2192} {}",
            fmt_shape(&left.shape),
            fmt_shape(&right.shape),
        ));
    }
    if dtype_changed {
        parts.push(format!("type {} \u{2192} {}", left.dtype, right.dtype));
    }
    if attrs_changed {
        let diff = right.num_attrs as i64 - left.num_attrs as i64;
        if diff > 0 {
            parts.push(format!(
                "{diff} attr{} added",
                if diff == 1 { "" } else { "s" }
            ));
        } else {
            let abs = diff.unsigned_abs();
            parts.push(format!(
                "{abs} attr{} removed",
                if abs == 1 { "" } else { "s" }
            ));
        }
    }

    node.summary = Some(capitalize(&parts.join("; ")));
    Some(node)
}

fn diff_group(
    logical_path: &str,
    _h5_path: &str,
    left: &GroupMeta,
    right: &GroupMeta,
) -> Option<DiffNode> {
    let attrs_changed = left.num_attrs != right.num_attrs;

    if !attrs_changed {
        return None;
    }

    let diff = right.num_attrs as i64 - left.num_attrs as i64;
    let summary = if diff > 0 {
        format!(
            "{diff} attribute{} added",
            if diff == 1 { "" } else { "s" }
        )
    } else {
        let abs = diff.unsigned_abs();
        format!(
            "{abs} attribute{} removed",
            if abs == 1 { "" } else { "s" }
        )
    };

    let mut node = DiffNode::new("modify", "hdf5_group", logical_path)
        .with_summary(capitalize(&summary))
        .with_detail("attrs_left", serde_json::json!(left.num_attrs))
        .with_detail("attrs_right", serde_json::json!(right.num_attrs));

    node.tags.insert("binoc-hdf5.attr-change".into());
    Some(node)
}

fn added_entry(file_logical: &str, h5_path: &str, meta: &EntryMeta) -> DiffNode {
    let path = h5_path_to_logical(file_logical, h5_path);
    match meta {
        EntryMeta::Dataset(ds) => DiffNode::new("add", "hdf5_dataset", &path)
            .with_summary(format!(
                "Dataset added {} {}",
                fmt_shape(&ds.shape),
                ds.dtype,
            ))
            .with_tag("binoc-hdf5.dataset-addition")
            .with_tag("binoc.schema-change")
            .with_detail("shape", serde_json::json!(ds.shape))
            .with_detail("dtype", serde_json::json!(&ds.dtype))
            .with_detail("num_attrs", serde_json::json!(ds.num_attrs)),
        EntryMeta::Group(gr) => DiffNode::new("add", "hdf5_group", &path)
            .with_summary(format!(
                "Group added ({} dataset{}, {} attr{})",
                gr.num_datasets,
                if gr.num_datasets == 1 { "" } else { "s" },
                gr.num_attrs,
                if gr.num_attrs == 1 { "" } else { "s" },
            ))
            .with_tag("binoc-hdf5.group-addition")
            .with_tag("binoc.schema-change"),
    }
}

fn removed_entry(file_logical: &str, h5_path: &str, meta: &EntryMeta) -> DiffNode {
    let path = h5_path_to_logical(file_logical, h5_path);
    match meta {
        EntryMeta::Dataset(ds) => DiffNode::new("remove", "hdf5_dataset", &path)
            .with_summary(format!(
                "Dataset removed {} {}",
                fmt_shape(&ds.shape),
                ds.dtype,
            ))
            .with_tag("binoc-hdf5.dataset-removal")
            .with_tag("binoc.schema-change")
            .with_detail("shape", serde_json::json!(ds.shape))
            .with_detail("dtype", serde_json::json!(&ds.dtype))
            .with_detail("num_attrs", serde_json::json!(ds.num_attrs)),
        EntryMeta::Group(gr) => DiffNode::new("remove", "hdf5_group", &path)
            .with_summary(format!(
                "Group removed ({} dataset{}, {} attr{})",
                gr.num_datasets,
                if gr.num_datasets == 1 { "" } else { "s" },
                gr.num_attrs,
                if gr.num_attrs == 1 { "" } else { "s" },
            ))
            .with_tag("binoc-hdf5.group-removal")
            .with_tag("binoc.schema-change"),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pair(left: &Path, right: &Path, logical: &str) -> ItemPair {
        ItemPair::both(
            Item::new(left.to_path_buf(), logical),
            Item::new(right.to_path_buf(), logical),
        )
    }

    fn create_test_file(path: &Path, setup: impl FnOnce(&hdf5::File)) {
        let file = hdf5::File::create(path).unwrap();
        setup(&file);
    }

    #[test]
    fn identical_files() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        let setup = |f: &hdf5::File| {
            let g = f.create_group("beam0000").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.0f64, 2.0, 3.0])
                .create("elevation")
                .unwrap();
        };
        create_test_file(&a, setup);
        create_test_file(&b, setup);

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = make_pair(&a, &b, "test.h5");
        let result = cmp.compare(&pair, &ctx).unwrap();
        assert!(matches!(result, CompareResult::Identical));
    }

    #[test]
    fn dataset_addition() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            let g = f.create_group("data").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.0f64, 2.0])
                .create("x")
                .unwrap();
        });
        create_test_file(&b, |f| {
            let g = f.create_group("data").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.0f64, 2.0])
                .create("x")
                .unwrap();
            g.new_dataset_builder()
                .with_data(&[3.0f64, 4.0])
                .create("y")
                .unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = make_pair(&a, &b, "test.h5");
        let result = cmp.compare(&pair, &ctx).unwrap();

        match result {
            CompareResult::Leaf(node) => {
                assert_eq!(node.kind, "modify");
                assert_eq!(node.item_type, "hdf5_file");
                assert_eq!(node.children.len(), 1);
                let child = &node.children[0];
                assert_eq!(child.kind, "add");
                assert_eq!(child.item_type, "hdf5_dataset");
                assert!(child.tags.contains("binoc-hdf5.dataset-addition"));
                assert_eq!(child.path, "test.h5/data/y");
            }
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn dataset_shape_change() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            f.new_dataset_builder()
                .with_data(&[1.0f64, 2.0, 3.0])
                .create("values")
                .unwrap();
        });
        create_test_file(&b, |f| {
            f.new_dataset_builder()
                .with_data(&[1.0f64, 2.0, 3.0, 4.0, 5.0])
                .create("values")
                .unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = make_pair(&a, &b, "test.h5");
        let result = cmp.compare(&pair, &ctx).unwrap();

        match result {
            CompareResult::Leaf(node) => {
                assert_eq!(node.kind, "modify");
                assert_eq!(node.children.len(), 1);
                let child = &node.children[0];
                assert_eq!(child.kind, "modify");
                assert_eq!(child.item_type, "hdf5_dataset");
                assert!(child.tags.contains("binoc-hdf5.shape-change"));
                assert_eq!(child.details["shape_left"], serde_json::json!([3]));
                assert_eq!(child.details["shape_right"], serde_json::json!([5]));
            }
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn group_addition() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            f.create_group("beam0000").unwrap();
        });
        create_test_file(&b, |f| {
            f.create_group("beam0000").unwrap();
            f.create_group("beam0001").unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = make_pair(&a, &b, "test.h5");
        let result = cmp.compare(&pair, &ctx).unwrap();

        match result {
            CompareResult::Leaf(node) => {
                assert_eq!(node.kind, "modify");
                assert_eq!(node.children.len(), 1);
                let child = &node.children[0];
                assert_eq!(child.kind, "add");
                assert_eq!(child.item_type, "hdf5_group");
                assert!(child.tags.contains("binoc-hdf5.group-addition"));
                assert_eq!(child.path, "test.h5/beam0001");
            }
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn group_removal() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            let g = f.create_group("sensors").unwrap();
            g.new_dataset_builder()
                .with_data(&[1i32, 2, 3])
                .create("temp")
                .unwrap();
        });
        create_test_file(&b, |_f| {
            // empty file
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = make_pair(&a, &b, "test.h5");
        let result = cmp.compare(&pair, &ctx).unwrap();

        match result {
            CompareResult::Leaf(node) => {
                assert_eq!(node.kind, "modify");
                // should have removed group + removed dataset
                assert!(node.children.len() >= 2);
                let kinds: Vec<&str> = node.children.iter().map(|c| c.kind.as_str()).collect();
                assert!(kinds.iter().all(|k| *k == "remove"));
            }
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn file_added() {
        let dir = tempfile::tempdir().unwrap();
        let b = dir.path().join("b.h5");

        create_test_file(&b, |f| {
            let g = f.create_group("data").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.0f64, 2.0])
                .create("values")
                .unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = ItemPair::added(Item::new(&b, "new.h5"));
        let result = cmp.compare(&pair, &ctx).unwrap();

        match result {
            CompareResult::Leaf(node) => {
                assert_eq!(node.kind, "add");
                assert_eq!(node.item_type, "hdf5_file");
                let summary = node.summary.unwrap();
                assert!(summary.contains("1 group"));
                assert!(summary.contains("1 dataset"));
            }
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn file_removed() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");

        create_test_file(&a, |f| {
            f.create_group("g1").unwrap();
            f.create_group("g2").unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = ItemPair::removed(Item::new(&a, "old.h5"));
        let result = cmp.compare(&pair, &ctx).unwrap();

        match result {
            CompareResult::Leaf(node) => {
                assert_eq!(node.kind, "remove");
                assert_eq!(node.item_type, "hdf5_file");
                assert!(node.summary.unwrap().contains("2 groups"));
            }
            _ => panic!("expected Leaf"),
        }
    }
}
