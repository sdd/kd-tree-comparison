use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use fixed::traits::LossyFrom;
use rand::distributions::{Distribution, Standard};
use std::ops::{Add, Mul};

use kd_tree_comparison::batch_benches;
use kiddo_v6::dist::{DistanceMetricCore, KdTreeDistanceMetric};
use kiddo_v6::kd_tree::leaf_strategies::VecOfArrays;
use kiddo_v6::kd_tree::leaf_view::TlsLeafScratch;
use kiddo_v6::kd_tree::KdTree;
use kiddo_v6::stem_strategies::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use kiddo_v6::stem_strategies::SimdPrune;
use kiddo_v6::traits::{Axis, Content};
use kiddo_v6::traits_unified_2::AxisUnified;
use kiddo_v6::{Eytzinger, SquaredEuclidean};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$a, $t, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("kiddo_v6_vecofarrays_eytzinger {}", $subtype),
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
        profile_sizes
    );

    batch_benches!(
        group,
        bench_float,
        [(f32, 2), (f32, 3), (f32, 4)],
        profile_sizes
    );

    group.finish();
}

fn bench_query_nearest_one_float<
    'a,
    A: Axis
        + AxisUnified<Coord = A>
        + LossyFrom<A>
        + SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + Add<Output = A>
        + Mul<Output = A>
        + 'static,
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
    SquaredEuclidean<A>: KdTreeDistanceMetric<A, K>,
    <SquaredEuclidean<A> as DistanceMetricCore<A>>::Output: AxisUnified<Coord = <SquaredEuclidean<A> as DistanceMetricCore<A>>::Output>
        + SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
{
    let mut points = vec![];
    points.resize_with(initial_size, || rand::random::<[A; K]>());

    let kdtree =
        KdTree::<A, T, Eytzinger<K>, VecOfArrays<A, T, K, BUCKET_SIZE>, K, BUCKET_SIZE>::new_from_slice(&points);

    let query_points: Vec<_> = (0..query_point_qty)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.iter().for_each(|point| {
                black_box(kdtree.nearest_one::<SquaredEuclidean<A>>(point));
            });
        });
    });
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
