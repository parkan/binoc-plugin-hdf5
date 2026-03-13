// compare two h5 files, optionally applying the rename detector transformer
use binoc_core::traits::{Comparator, CompareContext, Transformer};
use binoc_core::types::{Item, ItemPair, CompareResult, TransformResult};
use binoc_plugin_hdf5::{Hdf5Comparator, Hdf5RenameDetector};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: {} <left.h5> <right.h5> [--detect-renames]", args[0]);
        std::process::exit(1);
    }

    let detect_renames = args.get(3).map_or(false, |a| a == "--detect-renames");

    let left = Item::new(&args[1], &args[1]);
    let right = Item::new(&args[2], &args[2]);
    let pair = ItemPair::both(left, right);

    let cmp = Hdf5Comparator;
    let ctx = CompareContext::new();

    match cmp.compare(&pair, &ctx) {
        Ok(result) => {
            let node = match result {
                CompareResult::Identical => {
                    println!("Identical");
                    return;
                }
                CompareResult::Leaf(n) => n,
                CompareResult::Expand(n, _) => n,
            };

            let node = if detect_renames {
                let detector = Hdf5RenameDetector;
                match detector.transform(node, &ctx) {
                    TransformResult::Replace(n) => *n,
                    _ => { eprintln!("no renames detected"); return; }
                }
            } else {
                node
            };

            println!("{}", serde_json::to_string_pretty(&node).unwrap());
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
