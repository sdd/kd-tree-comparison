use std::num::NonZero;
use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rand::distributions::{Distribution, Standard};

use Kiddo_v5::float::distance::SquaredEuclidean;
use Kiddo_v5::float::kdtree::Axis;
use Kiddo_v5::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use Kiddo_v5::immutable::float::kdtree::ImmutableKdTree;
use Kiddo_v5::traits::Content;
use kiddo_v3::batch_benches;
use rayon::prelude::*;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_n_float_10::<$a, $t, $k>(
            &mut $group,
            $size,
            &format!("Kiddo_v5_immutable {}", $subtype),
        );
    };
}

pub fn nearest_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 10");
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

fn bench_query_nearest_n_float_10<
    'a,
    A: Axis + 'static + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content + 'static,
    const K: usize,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let initial_points: Vec<_> = (0..initial_size)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let kdtree = ImmutableKdTree::<A, T, K, BUCKET_SIZE>::new_from_slice(&initial_points);

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n_within::<SquaredEuclidean>(
                    point,
                    A::infinity(),
                    NonZero::new(10usize).unwrap(),
                    true,
                ));
            });
        });
    });
}

macro_rules! bench_float_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_n_float_100::<$a, $t, $k>(
            &mut $group,
            $size,
            &format!("Kiddo_v5_immutable {}", $subtype),
        );
    };
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

fn bench_query_nearest_n_float_100<'a, A: Axis + 'static, T: Content + 'static, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let initial_points: Vec<_> = (0..initial_size)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let kdtree = ImmutableKdTree::<A, T, K, BUCKET_SIZE>::new_from_slice(&initial_points);

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n_within::<SquaredEuclidean>(
                    point,
                    A::infinity(),
                    NonZero::new(100usize).unwrap(),
                    true,
                ));
            });
        });
    });
}

criterion_group!(benches, nearest_100);
criterion_main!(benches);
