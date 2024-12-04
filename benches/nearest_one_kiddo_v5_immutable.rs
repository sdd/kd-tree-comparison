use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

use kiddo_next::float::distance::SquaredEuclidean;
use kiddo_next::float::kdtree::Axis;
use kiddo_next::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use kiddo_next::immutable::float::kdtree::ImmutableKdTree;
use kiddo_next::traits::Content;
use kiddo_v3::batch_benches;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$a, $t, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("Kiddo_v5_immutable {}", $subtype),
        );
    };
}

pub fn nearest_one(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 1");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float,
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
        bench_float,
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

fn bench_query_nearest_one_float<
    'a,
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K> + 'static,
    T: Content + 'static,
    const K: usize,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let mut points = vec![];
    points.resize_with(initial_size, || rand::random::<[A; K]>());

    let kdtree = ImmutableKdTree::<A, T, K, BUCKET_SIZE>::new_from_slice(&points);

    let query_points: Vec<_> = (0..query_point_qty)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_one::<SquaredEuclidean>(point));
            });
        });
    });
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
