use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rayon::prelude::*;

use kiddo_v3::batch_benches;
use kiddo_v6::kd_tree::leaf_strategies::FlatVec;
use kiddo_v6::kd_tree::KdTree;
use kiddo_v6::stem_strategies::Eytzinger;
use kiddo_v6::traits_unified_2::SquaredEuclidean;
use std::num::NonZero;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {{
        #[allow(non_snake_case)]
        fn dispatch<A: 'static>(
            group: &mut BenchmarkGroup<WallTime>,
            size: usize,
            k: usize,
            subtype: String,
        ) {
            use std::any::TypeId;
            if TypeId::of::<A>() == TypeId::of::<f32>() {
                match k {
                    2 => {
                        bench_query_nearest_n_f32::<2>(group, size, QUERY_POINTS_PER_LOOP, &subtype)
                    }
                    3 => {
                        bench_query_nearest_n_f32::<3>(group, size, QUERY_POINTS_PER_LOOP, &subtype)
                    }
                    4 => {
                        bench_query_nearest_n_f32::<4>(group, size, QUERY_POINTS_PER_LOOP, &subtype)
                    }
                    _ => panic!("Unsupported K value"),
                }
            } else if TypeId::of::<A>() == TypeId::of::<f64>() {
                match k {
                    2 => {
                        bench_query_nearest_n_f64::<2>(group, size, QUERY_POINTS_PER_LOOP, &subtype)
                    }
                    3 => {
                        bench_query_nearest_n_f64::<3>(group, size, QUERY_POINTS_PER_LOOP, &subtype)
                    }
                    4 => {
                        bench_query_nearest_n_f64::<4>(group, size, QUERY_POINTS_PER_LOOP, &subtype)
                    }
                    _ => panic!("Unsupported K value"),
                }
            }
        }
        dispatch::<$a>(
            &mut $group,
            $size,
            $k,
            format!("Kiddo_v6_eytzinger_flatvec {}", $subtype),
        );
    }};
}

pub fn nearest_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 100");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float_100,
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

    batch_benches!(
        group,
        bench_float_100,
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

fn bench_query_nearest_n_f32<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) {
    use std::array;
    let mut points = vec![];
    points.resize_with(initial_size, || array::from_fn(|_| rand::random::<f32>()));

    let kdtree: KdTree<
        f32,
        usize,
        Eytzinger<K>,
        FlatVec<f32, usize, K, BUCKET_SIZE>,
        K,
        BUCKET_SIZE,
    > = KdTree::new_from_slice(&points);

    let query_points: Vec<_> = (0..query_point_qty)
        .map(|_| array::from_fn(|_| rand::random::<f32>()))
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n_within::<SquaredEuclidean<f32>>(
                    point,
                    f32::INFINITY,
                    NonZero::new(100).unwrap(),
                    true,
                ));
            });
        });
    });
}

fn bench_query_nearest_n_f64<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) {
    use std::array;
    let mut points = vec![];
    points.resize_with(initial_size, || array::from_fn(|_| rand::random::<f64>()));

    let kdtree: KdTree<
        f64,
        usize,
        Eytzinger<K>,
        FlatVec<f64, usize, K, BUCKET_SIZE>,
        K,
        BUCKET_SIZE,
    > = KdTree::new_from_slice(&points);

    let query_points: Vec<_> = (0..query_point_qty)
        .map(|_| array::from_fn(|_| rand::random::<f64>()))
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n_within::<SquaredEuclidean<f64>>(
                    point,
                    f64::INFINITY,
                    NonZero::new(100).unwrap(),
                    true,
                ));
            });
        });
    });
}

criterion_group!(benches, nearest_100);
criterion_main!(benches);
