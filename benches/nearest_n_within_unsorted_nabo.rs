use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use kiddo_v2::batch_benches;
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{AddAssign, SubAssign};

pub mod nabo_points;
use nabo::{CandidateContainer, KDTree};
use nabo_points::random_point_cloud;
use num_traits::Float;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;
const MAX_RESULTS: u32 = 32000;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_float::<$a, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("nabo {}", $subtype),
        );
    };
}

pub fn within(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query nearest n within radius unsorted");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float,
        [(f32, 2), (f64, 2), (f32, 3), (f64, 3), (f32, 4), (f64, 4)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32),
            (10_000_000, u32, u32)
        ]
    );

    group.finish();
}

fn bench_query_float<
    'a,
    A: Float + Debug + Default + AddAssign + SubAssign + Sync + Send,
    const K: usize,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[A; K]>,
    f64: Cast<A>,
{
    let points_to_add = random_point_cloud(initial_size as u32);

    let tree = KDTree::new_with_bucket_size(&points_to_add, BUCKET_SIZE as u32);

    let query_points = random_point_cloud(query_point_qty as u32);

    let params = nabo::Parameters {
        epsilon: A::zero(),

        // Nabo works with normalized radius, rather than dist according to the metric.
        // We need to convert RADIUS so that it performs the same query as the other libraries
        // being benchmarked
        max_radius: RADIUS.az::<A>().sqrt(),
        allow_self_match: true,
        sort_results: false,
    };

    let max_results_map = HashMap::from([
        (100usize, 3u32),
        (1_000, 10),
        (10_000, 100),
        (100_000, 100),
        (1_000_000, 100),
        (10_000_000, 1000),
    ]);

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                let max_results = *max_results_map.get(&initial_size).unwrap();
                // println!("max results for {} is {}", initial_size, max_results);
                black_box(tree.knn_advanced(
                    max_results,
                    &point,
                    CandidateContainer::BinaryHeap,
                    &params,
                    None,
                ));
            });
        });
    });
}

criterion_group!(benches, within);
criterion_main!(benches);
