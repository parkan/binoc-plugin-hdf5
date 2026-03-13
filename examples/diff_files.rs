// quick manual test: compare two h5 files passed as args
use binoc_core::traits::{Comparator, CompareContext};
use binoc_core::types::{Item, ItemPair};
use binoc_plugin_hdf5::Hdf5Comparator;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <left.h5> <right.h5>", args[0]);
        std::process::exit(1);
    }

    let left = Item::new(&args[1], &args[1]);
    let right = Item::new(&args[2], &args[2]);
    let pair = ItemPair::both(left, right);

    let cmp = Hdf5Comparator;
    let ctx = CompareContext::new();

    match cmp.compare(&pair, &ctx) {
        Ok(result) => {
            let json = match result {
                binoc_core::types::CompareResult::Identical => {
                    println!("Identical");
                    return;
                }
                binoc_core::types::CompareResult::Leaf(node) => {
                    serde_json::to_string_pretty(&node).unwrap()
                }
                binoc_core::types::CompareResult::Expand(node, _) => {
                    serde_json::to_string_pretty(&node).unwrap()
                }
            };
            println!("{json}");
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
