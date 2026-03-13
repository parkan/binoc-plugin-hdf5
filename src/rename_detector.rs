use std::collections::BTreeSet;

use binoc_core::ir::DiffNode;
use binoc_core::traits::{CompareContext, Transformer};
use binoc_core::types::*;

use crate::hdf5_compare::{EntryMeta, Hdf5DataPair};

/// Detects renamed HDF5 groups by fingerprinting their internal structure.
/// When a removed group and an added group have identical child datasets
/// (same relative names, shapes, dtypes), rewrites the pair into a single
/// rename node. Requires the cached Hdf5DataPair from the comparator --
/// individual add/remove DiffNodes only carry one side's data.
pub struct Hdf5RenameDetector;

// structural fingerprint of a group: the set of its child datasets
// identified by (name, shape, dtype, storage_size)
type GroupFingerprint = BTreeSet<(String, Vec<usize>, String, u64)>;

fn fingerprint_group(
    group_path: &str,
    inventory: &std::collections::BTreeMap<String, EntryMeta>,
) -> GroupFingerprint {
    let prefix = format!("{group_path}/");
    inventory
        .iter()
        .filter_map(|(path, meta)| {
            // only direct children (one level below group_path)
            let suffix = path.strip_prefix(&prefix)?;
            if suffix.contains('/') {
                return None;
            }
            match meta {
                EntryMeta::Dataset(ds) => Some((
                    suffix.to_string(),
                    ds.shape.clone(),
                    ds.dtype.clone(),
                    ds.storage_size,
                )),
                _ => None,
            }
        })
        .collect()
}

impl Transformer for Hdf5RenameDetector {
    fn name(&self) -> &str {
        "binoc-hdf5.rename_detector"
    }

    fn match_types(&self) -> &[&str] {
        &["hdf5_file"]
    }

    fn scope(&self) -> TransformScope {
        TransformScope::Subtree
    }

    fn transform(&self, mut node: DiffNode, ctx: &CompareContext) -> TransformResult {
        let cached = ctx.get_cached_data(&node.path);
        let Some(ReopenedData::Custom(boxed)) = cached.as_ref() else {
            return TransformResult::Unchanged;
        };
        let Some(pair) = boxed.as_any().downcast_ref::<Hdf5DataPair>() else {
            return TransformResult::Unchanged;
        };
        let (Some(left_inv), Some(right_inv)) = (&pair.left, &pair.right) else {
            return TransformResult::Unchanged;
        };

        // collect removed and added groups
        let mut removed_groups: Vec<usize> = Vec::new();
        let mut added_groups: Vec<usize> = Vec::new();

        for (i, child) in node.children.iter().enumerate() {
            if child.item_type == "hdf5_group" {
                match child.kind.as_str() {
                    "remove" => removed_groups.push(i),
                    "add" => added_groups.push(i),
                    _ => {}
                }
            }
        }

        if removed_groups.is_empty() || added_groups.is_empty() {
            return TransformResult::Unchanged;
        }

        // fingerprint removed groups from left inventory, added from right
        let removed_fps: Vec<(usize, String, GroupFingerprint)> = removed_groups
            .iter()
            .filter_map(|&i| {
                let child = &node.children[i];
                let h5_path = find_h5_path(left_inv, &child.path)?;
                let fp = fingerprint_group(&h5_path, left_inv);
                if fp.is_empty() {
                    return None;
                }
                Some((i, h5_path, fp))
            })
            .collect();

        let added_fps: Vec<(usize, String, GroupFingerprint)> = added_groups
            .iter()
            .filter_map(|&i| {
                let child = &node.children[i];
                let h5_path = find_h5_path(right_inv, &child.path)?;
                let fp = fingerprint_group(&h5_path, right_inv);
                if fp.is_empty() {
                    return None;
                }
                Some((i, h5_path, fp))
            })
            .collect();

        // match by fingerprint
        let mut matched_removed: BTreeSet<usize> = BTreeSet::new();
        let mut matched_added: BTreeSet<usize> = BTreeSet::new();
        let mut renames: Vec<DiffNode> = Vec::new();

        for &(ri, ref r_h5, ref r_fp) in &removed_fps {
            if matched_removed.contains(&ri) {
                continue;
            }
            for &(ai, ref a_h5, ref a_fp) in &added_fps {
                if matched_added.contains(&ai) {
                    continue;
                }
                if r_fp == a_fp {
                    matched_removed.insert(ri);
                    matched_added.insert(ai);

                    let r_name = r_h5.rsplit('/').next().unwrap_or(r_h5);
                    let a_name = a_h5.rsplit('/').next().unwrap_or(a_h5);

                    let rename_node = DiffNode::new("rename", "hdf5_group", &node.children[ai].path)
                        .with_source_path(&node.children[ri].path)
                        .with_summary(format!("{r_name} \u{2192} {a_name} (structure unchanged)"))
                        .with_tag("binoc-hdf5.group-rename")
                        .with_detail(
                            "datasets",
                            serde_json::json!(r_fp.iter().map(|(n, _, _, _)| n).collect::<Vec<_>>()),
                        );
                    renames.push(rename_node);
                    break;
                }
            }
        }

        if renames.is_empty() {
            return TransformResult::Unchanged;
        }

        // also remove child datasets that belonged to matched groups
        let removed_group_paths: BTreeSet<String> = matched_removed
            .iter()
            .map(|&i| node.children[i].path.clone())
            .collect();
        let added_group_paths: BTreeSet<String> = matched_added
            .iter()
            .map(|&i| node.children[i].path.clone())
            .collect();

        let mut new_children: Vec<DiffNode> = Vec::new();
        for (i, child) in node.children.into_iter().enumerate() {
            if matched_removed.contains(&i) || matched_added.contains(&i) {
                continue;
            }
            // skip datasets that were children of matched groups
            let is_child_of_matched = removed_group_paths
                .iter()
                .any(|gp| child.path.starts_with(gp.as_str()) && child.path.len() > gp.len())
                || added_group_paths
                    .iter()
                    .any(|gp| child.path.starts_with(gp.as_str()) && child.path.len() > gp.len());
            if is_child_of_matched {
                continue;
            }
            new_children.push(child);
        }
        new_children.extend(renames);

        node.children = new_children;
        TransformResult::Replace(Box::new(node))
    }
}

fn find_h5_path(
    inventory: &std::collections::BTreeMap<String, EntryMeta>,
    node_path: &str,
) -> Option<String> {
    inventory
        .keys()
        .find(|h5_path| {
            let trimmed = h5_path.trim_start_matches('/');
            !trimmed.is_empty() && node_path.ends_with(trimmed)
        })
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use binoc_core::traits::Comparator;
    use crate::Hdf5Comparator;
    use std::path::Path;

    fn create_test_file(path: &Path, setup: impl FnOnce(&hdf5::File)) {
        let file = hdf5::File::create(path).unwrap();
        setup(&file);
    }

    fn make_beam(file: &hdf5::File, name: &str) {
        let g = file.create_group(name).unwrap();
        g.new_dataset_builder()
            .with_data(&[1.0f64, 2.0, 3.0])
            .create("elevation")
            .unwrap();
        g.new_dataset_builder()
            .with_data(&[10.0f64, 20.0, 30.0])
            .create("latitude")
            .unwrap();
    }

    #[test]
    fn detects_beam_renames() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            make_beam(f, "BEAM0000");
            make_beam(f, "BEAM0001");
        });
        create_test_file(&b, |f| {
            make_beam(f, "BEAM0101");
            make_beam(f, "BEAM0110");
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = binoc_core::types::ItemPair::both(
            binoc_core::types::Item::new(&a, "test.h5"),
            binoc_core::types::Item::new(&b, "test.h5"),
        );

        // comparator produces adds + removes
        let result = cmp.compare(&pair, &ctx).unwrap();
        let node = match result {
            CompareResult::Leaf(n) => n,
            _ => panic!("expected Leaf"),
        };

        // without transformer: 2 removed groups + 2 added groups + 4 removed datasets + 4 added datasets
        let adds = node.children.iter().filter(|c| c.kind == "add").count();
        let removes = node.children.iter().filter(|c| c.kind == "remove").count();
        assert!(adds >= 2);
        assert!(removes >= 2);
        let total_before = node.children.len();

        // apply transformer
        let detector = Hdf5RenameDetector;
        let result = detector.transform(node, &ctx);
        let node = match result {
            TransformResult::Replace(n) => *n,
            other => panic!("expected Replace, got {:?}", std::mem::discriminant(&other)),
        };

        // after transformer: 2 rename nodes, no add/remove for matched groups/datasets
        let renames: Vec<&DiffNode> = node
            .children
            .iter()
            .filter(|c| c.kind == "rename")
            .collect();
        assert_eq!(renames.len(), 2, "expected 2 renames, got {}", renames.len());

        // verify rename details
        for r in &renames {
            assert_eq!(r.item_type, "hdf5_group");
            assert!(r.tags.contains("binoc-hdf5.group-rename"));
            assert!(r.source_path.is_some());
            assert!(r.summary.as_ref().unwrap().contains("\u{2192}"));
        }

        // total children should be much smaller
        assert!(
            node.children.len() < total_before,
            "expected fewer children after rename detection: {} vs {}",
            node.children.len(),
            total_before,
        );
    }

    #[test]
    fn no_match_when_structure_differs() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            let g = f.create_group("sensors").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.0f64, 2.0])
                .create("temp")
                .unwrap();
        });
        create_test_file(&b, |f| {
            let g = f.create_group("probes").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.0f32, 2.0, 3.0]) // different dtype AND shape
                .create("pressure")             // different name
                .unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = binoc_core::types::ItemPair::both(
            binoc_core::types::Item::new(&a, "test.h5"),
            binoc_core::types::Item::new(&b, "test.h5"),
        );

        let result = cmp.compare(&pair, &ctx).unwrap();
        let node = match result {
            CompareResult::Leaf(n) => n,
            _ => panic!("expected Leaf"),
        };

        let detector = Hdf5RenameDetector;
        let result = detector.transform(node, &ctx);

        // no renames possible -- structure is completely different
        assert!(matches!(result, TransformResult::Unchanged));
    }

    #[test]
    fn partial_rename_with_other_changes() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.h5");
        let b = dir.path().join("b.h5");

        create_test_file(&a, |f| {
            make_beam(f, "BEAM0000");
            let g = f.create_group("metadata").unwrap();
            g.new_dataset_builder()
                .with_data(&[42i32])
                .create("version")
                .unwrap();
        });
        create_test_file(&b, |f| {
            make_beam(f, "BEAM0101"); // renamed from BEAM0000
            let g = f.create_group("metadata").unwrap();
            g.new_dataset_builder()
                .with_data(&[43i32]) // different shape (1 element but value differs -- same shape though)
                .create("version")
                .unwrap();
            // new group not matching anything
            f.create_group("quality_flags").unwrap();
        });

        let cmp = Hdf5Comparator;
        let ctx = CompareContext::new();
        let pair = binoc_core::types::ItemPair::both(
            binoc_core::types::Item::new(&a, "test.h5"),
            binoc_core::types::Item::new(&b, "test.h5"),
        );

        let result = cmp.compare(&pair, &ctx).unwrap();
        let node = match result {
            CompareResult::Leaf(n) => n,
            _ => panic!("expected Leaf"),
        };

        let detector = Hdf5RenameDetector;
        let result = detector.transform(node, &ctx);
        let node = match result {
            TransformResult::Replace(n) => *n,
            _ => panic!("expected Replace"),
        };

        // should have exactly 1 rename (BEAM0000 -> BEAM0101)
        let renames: Vec<&DiffNode> = node
            .children
            .iter()
            .filter(|c| c.kind == "rename")
            .collect();
        assert_eq!(renames.len(), 1);

        // quality_flags should still be an add
        let adds: Vec<&DiffNode> = node
            .children
            .iter()
            .filter(|c| c.kind == "add")
            .collect();
        assert!(adds.iter().any(|c| c.path.contains("quality_flags")));
    }
}
