use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rayon::prelude::*;

use kiddo_v3::batch_benches;
use kiddo_v6::kd_tree::leaf_strategies::VecOfArrays;
use kiddo_v6::kd_tree::KdTree;
use kiddo_v6::stem_strategies::{Block3, DonnellyMarkerSimd};
use kiddo_v6::traits_unified_2::SquaredEuclidean;
use std::num::NonZero;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {{
        // Use a helper function to dispatch at runtime based on type
        #[allow(non_snake_case)]
        fn dispatch<A: 'static>(
            group: &mut BenchmarkGroup<WallTime>,
            size: usize,
            k: usize,
            subtype: String,
        ) {
            use std::any::TypeId;
            if TypeId::of::<A>() == TypeId::of::<f64>() {
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
            // Block3 SIMD is only valid for f64, so we skip f32
        }
        dispatch::<$a>(
            &mut $group,
            $size,
            $k,
            format!("Kiddo_v6_donnelly_vecofarrays_simd_block3 {}", $subtype),
        );
    }};
}

pub fn nearest_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 100");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    // Block3 SIMD is only for f64
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

    group.finish();
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

    // DonnellyMarkerSimd Block3 for f64: N_LEVEL_1_SUBTREES=64, N_LEVEL_2_SUBTREES=8
    let kdtree: KdTree<
        f64,
        usize,
        DonnellyMarkerSimd<Block3, 64, 8, K>,
        VecOfArrays<f64, usize, K, BUCKET_SIZE>,
        K,
        BUCKET_SIZE,
    > = KdTree::new_from_slice(&points);

    let query_points: Vec<_> = (0..query_point_qty)
        .map(|_| array::from_fn(|_| rand::random::<f64>()))
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n::<SquaredEuclidean<f64>>(
                    point,
                    NonZero::new(100).unwrap(),
                    true,
                ));
            });
        });
    });
}

criterion_group!(benches, nearest_n);
criterion_main!(benches);
