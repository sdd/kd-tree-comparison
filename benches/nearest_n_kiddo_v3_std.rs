use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use fixed::types::extra::{Unsigned, U16};
use fixed::FixedU16;
use rand::distributions::{Distribution, Standard};

use kiddo_v3::batch_benches;
use kiddo_v3::fixed::distance::SquaredEuclidean as SquaredEuclideanFixed;
use kiddo_v3::fixed::kdtree::{Axis as AxisFixed, KdTree as FixedKdTree};
use kiddo_v3::float_leaf_simd::leaf_node::BestFromDists;
use kiddo_v3::float::distance::SquaredEuclidean;
use kiddo_v3::float::kdtree::{Axis, KdTree};
use kiddo_v3::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
use kiddo_v3::types::{Content, Index};
use rayon::prelude::*;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_n_float_10::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v3_std {}", $subtype),
        );
    };
}

macro_rules! bench_fixed_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_n_fixed_10::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v3_std {}", $subtype),
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
        [(f32, 2), (f64, 2), (f32, 3), (f64, 3), (f32, 4), (f64, 4)],
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
        bench_fixed_10,
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

    group.finish();
}

fn bench_query_nearest_n_float_10<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    A: BestFromDists<T, 32>,
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let mut kdtree = KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(initial_size);

    for _ in 0..initial_size {
        let point = rand::random::<([A; K], T)>();
        kdtree.add(&point.0, point.1);
    }

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n::<SquaredEuclidean>(point, 10));
            });
        });
    });
}

fn bench_query_nearest_n_fixed_10<
    'a,
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
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

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand_data_fixed_u16_point::<A, K>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n::<SquaredEuclideanFixed>(point, 10));
            });
        });
    });
}

macro_rules! bench_float_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_n_float_100::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v3_std {}", $subtype),
        );
    };
}

macro_rules! bench_fixed_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_n_fixed_100::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v3_std {}", $subtype),
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
        [(f32, 2), (f64, 2), (f32, 3), (f64, 3), (f32, 4), (f64, 4)],
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
        bench_fixed_100,
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

    group.finish();
}

fn bench_query_nearest_n_float_100<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    A: BestFromDists<T, 32>,
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let mut kdtree = KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(initial_size);

    for _ in 0..initial_size {
        let point = rand::random::<([A; K], T)>();
        kdtree.add(&point.0, point.1);
    }

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n::<SquaredEuclidean>(point, 100));
            });
        });
    });
}

fn bench_query_nearest_n_fixed_100<
    'a,
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
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

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand_data_fixed_u16_point::<A, K>())
        .collect();

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(kdtree.nearest_n::<SquaredEuclideanFixed>(point, 100));
            });
        });
    });
}

criterion_group!(benches, nearest_10, nearest_100);
criterion_main!(benches);
