use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use fixed::types::extra::{Unsigned, U16};
use fixed::FixedU16;
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

use kiddo_v3::batch_benches;
use kiddo_v3::float::distance::SquaredEuclidean;
use kiddo_v3::fixed::distance::SquaredEuclidean as SquaredEuclideanFixed;

use kiddo_v3::fixed::kdtree::{Axis as AxisFixed, KdTree as FixedKdTree};
use kiddo_v3::float::kdtree::{Axis,KdTree};
use kiddo_v3::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
use kiddo_v3::types::{Content, Index};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("Kiddo_v3_std {}", $subtype),
        );
    };
}

macro_rules! bench_empty_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_fixed::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("Kiddo_v3_std {}", $subtype),
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
        bench_empty_fixed,
        [(FXP, 2), (FXP, 3), (FXP, 4)],
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
        [(f32, 2), (f64, 2), (f64, 3), (f64, 4), (f32, 4), (f32, 3)],
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

fn bench_query_nearest_one_float<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let mut kdtree = KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(initial_size);

    for _ in 0..initial_size {
        let point = rand::random::<([A; K], T)>();
        kdtree.add(&point.0, point.1);
    }

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

fn bench_query_nearest_one_fixed<
    'a,
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    let mut kdtree =
        FixedKdTree::<FixedU16<A>, T, K, BUCKET_SIZE, IDX>::with_capacity(initial_size);

    for _ in 0..initial_size {
        let entry = rand_data_fixed_u16_entry::<A, T, K>();
        kdtree.add(&entry.0, entry.1);
    }

    let query_points: Vec<_> = (0..query_point_qty)
        .into_iter()
        .map(|_| rand_data_fixed_u16_point::<A, K>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_one::<SquaredEuclideanFixed>(point));
            });
        });
    });
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
