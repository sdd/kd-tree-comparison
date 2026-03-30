use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use std::collections::HashMap;
use std::num::NonZeroUsize;

use kiddo_v3::batch_benches_parameterized;
use kiddo_v6::kd_tree::leaf_strategies::FlatVec;
use kiddo_v6::kd_tree::KdTree;
use kiddo_v6::stem_strategies::Donnelly;
use kiddo_v6::traits_unified_2::SquaredEuclidean;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt, $subtype: expr) => {{
        // Use a helper function to dispatch at runtime based on type
        #[allow(non_snake_case)]
        fn dispatch<A: 'static, const K: usize>(
            group: &mut BenchmarkGroup<WallTime>,
            size: usize,
            radius: f64,
            subtype: String,
        ) where
            f64: Cast<A>,
            Standard: Distribution<A>,
        {
            use std::any::TypeId;
            if TypeId::of::<A>() == TypeId::of::<f32>() {
                bench_query_float_f32::<K>(group, size, radius, &subtype)
            } else if TypeId::of::<A>() == TypeId::of::<f64>() {
                bench_query_float_f64::<K>(group, size, radius, &subtype)
            }
        }
        dispatch::<$a, $k>(
            &mut $group,
            $size,
            $radius,
            format!("Kiddo_v6_donnelly_flatvec {}", $subtype),
        );
    }};
}

fn within(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query nearest n within radius");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches_parameterized!(
        group,
        bench_float,
        RADIUS,
        [(f64, 2), (f64, 3), (f64, 4)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32),
            (10_000_000, u32, u32)
        ]
    );

    batch_benches_parameterized!(
        group,
        bench_float,
        RADIUS,
        [(f32, 2), (f32, 3), (f32, 4)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16)
        ]
    );

    group.finish();
}

fn bench_query_float_f32<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) {
    use std::array;
    let mut points = vec![];
    points.resize_with(initial_size, || array::from_fn(|_| rand::random::<f32>()));

    let kdtree: KdTree<
        f32,
        usize,
        Donnelly<4, 64, 4, K>,
        FlatVec<f32, usize, K, BUCKET_SIZE>,
        K,
        BUCKET_SIZE,
    > = KdTree::new_from_slice(&points);

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .map(|_| array::from_fn(|_| rand::random::<f32>()))
        .collect();

    let max_results_map = HashMap::from([
        (100usize, 3usize),
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

                black_box(kdtree.nearest_n_within::<SquaredEuclidean<f32>>(
                    point,
                    radius.az::<f32>(),
                    NonZeroUsize::new(max_results).unwrap(),
                    true,
                ));
            });
        });
    });
}

fn bench_query_float_f64<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) {
    use std::array;
    let mut points = vec![];
    points.resize_with(initial_size, || array::from_fn(|_| rand::random::<f64>()));

    let kdtree: KdTree<
        f64,
        usize,
        Donnelly<3, 64, 8, K>,
        FlatVec<f64, usize, K, BUCKET_SIZE>,
        K,
        BUCKET_SIZE,
    > = KdTree::new_from_slice(&points);

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .map(|_| array::from_fn(|_| rand::random::<f64>()))
        .collect();

    let max_results_map = HashMap::from([
        (100usize, 3usize),
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

                black_box(kdtree.nearest_n_within::<SquaredEuclidean<f64>>(
                    point,
                    radius,
                    NonZeroUsize::new(max_results).unwrap(),
                    true,
                ));
            });
        });
    });
}

criterion_group!(benches, within);
criterion_main!(benches);
