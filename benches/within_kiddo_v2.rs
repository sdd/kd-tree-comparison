use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use fixed::types::extra::{LeEqU16, Unsigned, U16};
use fixed::FixedU16;
use rand::distributions::{Distribution, Standard};

use kiddo_v2::batch_benches_parameterized;
use kiddo_v2::distance::squared_euclidean;
use kiddo_v2::fixed::distance::squared_euclidean as squared_euclidean_fixedpoint;
use kiddo_v2::fixed::kdtree::{Axis as AxisFixed, KdTree as KdTreeFixed};
use kiddo_v2::float::kdtree::{Axis, KdTree};
use kiddo_v2::test_utils::{
    build_populated_tree_and_query_points_fixed, build_populated_tree_and_query_points_float,
    process_queries_fixed_parameterized, process_queries_float_parameterized,
};
use kiddo_v2::types::{Content, Index};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt,  $subtype: expr) => {
        bench_query_float::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            $radius,
            &format!("Kiddo_v2 {}", $subtype),
        );
    };
}

macro_rules! bench_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $radius:tt, $subtype: expr) => {
        bench_query_fixed::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            $radius,
            &format!("Kiddo_v2 {}", $subtype),
        );
    };
}

fn within(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query: within radius");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches_parameterized!(
        group,
        bench_float,
        RADIUS,
        [(f32, 2), (f64, 2), (f64, 3), (f64, 4), (f32, 3)],
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
        bench_fixed,
        RADIUS,
        [(FXP, 2), (FXP, 3)],
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

fn perform_query_float<
    A: Axis,
    T: Content + 'static,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX> + 'static,
>(
    kdtree: &KdTree<A, T, K, BUCKET_SIZE, IDX>,
    point: &[A; K],
    radius: f64,
) where
    usize: Cast<IDX>,
    f64: Cast<A>,
{
    let _res = kdtree.within(&point, radius.az::<A>(), &squared_euclidean);
}

fn perform_query_fixed<
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX> + 'static,
>(
    kdtree: &KdTreeFixed<FixedU16<A>, T, K, BUCKET_SIZE, IDX>,
    point: &[FixedU16<A>; K],
    radius: f64,
) where
    usize: Cast<IDX>,
    FixedU16<A>: AxisFixed,
    A: LeEqU16,
{
    let _res = kdtree.within(
        &point,
        FixedU16::<A>::from_num(radius),
        &squared_euclidean_fixedpoint,
    );
}

fn bench_query_float<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) where
    usize: Cast<IDX>,
    f64: Cast<A>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_float::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        QUERY_POINTS_PER_LOOP,
                    )
                },
                process_queries_float_parameterized(
                    perform_query_float::<A, T, K, BUCKET_SIZE, IDX>,
                    radius,
                ),
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_query_fixed<
    'a,
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    FixedU16<A>: AxisFixed,
    A: LeEqU16,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_fixed::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        QUERY_POINTS_PER_LOOP,
                    )
                },
                process_queries_fixed_parameterized(
                    perform_query_fixed::<A, T, K, BUCKET_SIZE, IDX>,
                    radius,
                ),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, within);
criterion_main!(benches);
