use std::num::NonZero;
use az::{Az, Cast};
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
use kiddo_v3::batch_benches_parameterized;

use rayon::prelude::*;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt,  $subtype: expr) => {
        bench_query_float::<$a, $t, $k>(
            &mut $group,
            $size,
            $radius,
            &format!("Kiddo_v5_immutable {}", $subtype),
        );
    };
}

fn within_unsorted(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query within radius unsorted");
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

fn bench_query_float<'a, A: Axis + 'static, T: Content + 'static, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) where
    A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    usize: Cast<T>,
    f64: Cast<A>,
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
                    radius.az::<A>(),
                    NonZero::new(usize::MAX).unwrap(),
                    false,
                ));
            });
        });
    });
}

criterion_group!(benches, within_unsorted);
criterion_main!(benches);
