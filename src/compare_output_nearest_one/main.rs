use std::collections::{HashMap, HashSet};
use std::error::Error;

use indicatif::ProgressBar;

use fnntw::Tree;
use kiddo_v5::float::distance::SquaredEuclidean;
use kiddo_v5::immutable::float::kdtree::ImmutableKdTree;
use kiddo_v5::float::kdtree::KdTree;
use nabo::{KDTree, NotNan};
use kd_tree_comparison::utils::nabo_points::P;

const BUCKET_SIZE: usize = 32;
const NUM_POINTS: usize = 10_000_000;
const NUM_QUERIES: usize = 10_000;

type A = f64;
const K: usize = 4; // dimensionality

#[cfg(not(tarpaulin_include))]
fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();

    println!("Generating data points...");
    let data_points: Vec<_> = (0..NUM_POINTS)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    println!("Generating query points...");
    let query_points: Vec<_> = (0..NUM_QUERIES)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    println!("Building Kiddo v3 immutable tree...");
    let kiddo_v3_immutable = ImmutableKdTree::<A, usize, K, BUCKET_SIZE>::new_from_slice(&data_points);

    println!("Building Kiddo v3 std tree...");
    let mut kiddo_v3_std = KdTree::<A, u32, K, BUCKET_SIZE, u32>::with_capacity(NUM_POINTS);
    for (idx, point) in data_points.iter().enumerate() {
        kiddo_v3_std.add(point, idx as u32);
    }

    println!("Building Nabo tree...");
    let nabo_point_structs: Vec<_> = data_points.iter().map(|point| {
        let mut res: [NotNan<A>; K] = [NotNan::new(0f64).unwrap(); K];
        for i in 0..K {
            res[i] = NotNan::new(point[i]).unwrap();
        }
        P(res)
    }).collect();
    let nabo = KDTree::new_with_bucket_size(&nabo_point_structs, BUCKET_SIZE as u32);

    println!("Building FNNTW tree...");
    let fnntw = Tree::new(&data_points, BUCKET_SIZE).unwrap();

    println!("Performing queries...");
    let bar = ProgressBar::new(NUM_QUERIES as u64);
    for (idx, query_point) in query_points.iter().enumerate() {
        let mut result_map: HashMap<usize, HashSet<String>> = HashMap::new();

        let res = kiddo_v3_immutable.nearest_one::<SquaredEuclidean>(query_point);
        result_map.entry(res.item).or_default().insert("KIDDO_V3_IMMUTABLE".to_string());

        let res = kiddo_v3_std.nearest_one::<SquaredEuclidean>(query_point);
        result_map.entry(res.item as usize).or_default().insert("KIDDO_V3_STD".to_string());

        let mut nabo_query_point: [NotNan<A>; K] = [NotNan::new(0f64).unwrap(); K];
        for i in 0..K {
            nabo_query_point[i] = NotNan::new(query_point[i]).unwrap();
        }
        let res = nabo.knn(1, & P(nabo_query_point));
        result_map.entry(res[0].index as usize).or_default().insert("NABO".to_string());

        let res = fnntw.query_nearest_k(query_point, 1).unwrap();
        result_map.entry(res[0].1 as usize).or_default().insert("KIDDO_V3_IMMUTABLE".to_string());

        if result_map.len() > 1 {
            println!("Discrepancy for query point #{}: {:?}", idx, &result_map);
        }
        bar.inc(1);
    }
    bar.finish();

    println!("Check complete");
    Ok(())
}
