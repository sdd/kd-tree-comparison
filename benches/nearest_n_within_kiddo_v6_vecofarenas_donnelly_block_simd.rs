use std::num::NonZero;
use std::ops::{Add, Mul};

use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use fixed::traits::LossyFrom;
use rand::distributions::{Distribution, Standard};

use kd_tree_comparison::batch_benches_parameterized;
use kiddo_v6::dist::{DistanceMetricCore, KdTreeDistanceMetric};
use kiddo_v6::kd_tree::leaf_strategies::VecOfArenas;
use kiddo_v6::kd_tree::leaf_view::TlsLeafScratch;
use kiddo_v6::kd_tree::KdTree;
use kiddo_v6::stem_strategies::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use kiddo_v6::stem_strategies::{Block3, Block4, DonnellyMarkerSimd, SimdPrune};
use kiddo_v6::traits::{Axis, Content};
use kiddo_v6::traits_unified_2::AxisUnified;
use kiddo_v6::{SquaredEuclidean, StemStrategy};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;

macro_rules! bench_f64 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt, $subtype: expr) => {
        bench_query_float::<f64, $t, DonnellyMarkerSimd<Block3, 64, 8, $k>, $k>(
            &mut $group,
            $size,
            $radius,
            &format!("kiddo_v6_vecofarenas_donnelly_block_simd {}", $subtype),
        );
    };
}

macro_rules! bench_f32 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt, $subtype: expr) => {
        bench_query_float::<f32, $t, DonnellyMarkerSimd<Block4, 64, 4, $k>, $k>(
            &mut $group,
            $size,
            $radius,
            &format!("kiddo_v6_vecofarenas_donnelly_block_simd {}", $subtype),
        );
    };
}

fn within(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query nearest n within radius");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches_parameterized!(
        group,
        bench_f64,
        RADIUS,
        [(f64, 2), (f64, 3), (f64, 4)],
        profile_sizes
    );

    batch_benches_parameterized!(
        group,
        bench_f32,
        RADIUS,
        [(f32, 2), (f32, 3), (f32, 4)],
        profile_sizes
    );

    group.finish();
}

fn bench_query_float<
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
    S: StemStrategy + 'static,
    const K: usize,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) where
    usize: Cast<T>,
    f64: Cast<A>,
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
    let initial_points: Vec<_> = (0..initial_size)
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let kdtree =
        KdTree::<A, T, S, VecOfArenas<A, T, K, BUCKET_SIZE>, K, BUCKET_SIZE>::new_from_slice(
            &initial_points,
        );

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let max_results = NonZero::new(kd_tree_comparison::nearest_n_within_max_results(
        initial_size,
    ))
    .unwrap();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.iter().for_each(|point| {
                black_box(kdtree.nearest_n_within::<SquaredEuclidean<A>>(
                    point,
                    <SquaredEuclidean<A> as DistanceMetricCore<A>>::widen_coord(radius.az::<A>()),
                    max_results,
                    true,
                ));
            });
        });
    });
}

criterion_group!(benches, within);
criterion_main!(benches);
