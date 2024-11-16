use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rand::distributions::{Distribution, Standard};

use kiddo_next::float::distance::SquaredEuclidean;
use kiddo_next::float::kdtree::Axis;
use kiddo_next::float_leaf_slice::leaf_slice::LeafSliceFloat;
use kiddo_next::immutable_dynamic::float::kdtree::ImmutableDynamicKdTree;
use kiddo_next::types::Content;
use kiddo_v3::batch_benches;
// use kiddo_v3::test_utils::{build_populated_tree_and_query_points_immutable_float, process_queries_immutable_float};
use rayon::prelude::*;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_best_n_float_10::<$a, $t, $k>(
            &mut $group,
            $size,
            &format!("Kiddo_v5_immutable_dynamic {}", $subtype),
        );
    };
}

pub fn best_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Best 10");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float_10,
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
        bench_float_10,
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

fn bench_query_best_n_float_10<'a, A: Axis + 'static, T: Content + 'static, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    usize: Cast<T>,
    f64: Cast<A>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
    A: LeafSliceFloat<T, K>,
{
    let initial_points: Vec<_> = (0..initial_size)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let kdtree = ImmutableDynamicKdTree::<A, T, K, BUCKET_SIZE>::new_from_slice(&initial_points);

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(
                    kdtree
                        .best_n_within::<SquaredEuclidean>(point, 0.05f64.az::<A>(), 10)
                        .min(),
                );
            });
        });
    });
}

criterion_group!(benches, best_10);
criterion_main!(benches);
