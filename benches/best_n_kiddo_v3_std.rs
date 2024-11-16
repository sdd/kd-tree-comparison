use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use fixed::traits::Fixed;
use fixed::types::extra::{Unsigned, U16};
use fixed::FixedU16;
use rand::distributions::{Distribution, Standard};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;

use kiddo_v3::batch_benches;
use kiddo_v3::fixed::distance::SquaredEuclidean as SquaredEuclideanFixed;
use kiddo_v3::fixed::kdtree::{Axis as AxisFixed, KdTree as KdTreeFixed};
use kiddo_v3::float::distance::SquaredEuclidean;
use kiddo_v3::float::kdtree::{Axis, KdTree};
use kiddo_v3::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
use kiddo_v3::types::{Content, Index};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_best_n_float_10::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v3_std {}", $subtype),
        );
    };
}

macro_rules! bench_fixed_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_best_n_fixed_10::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v3_std {}", $subtype),
        );
    };
}

pub fn best_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query: Best 10");
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
            (1_000_000, u32, u32)
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
            (1_000_000, u32, u32)
        ]
    );

    group.finish();
}

fn bench_query_best_n_float_10<
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
    usize: Cast<IDX>,
    f64: Cast<A>,
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
                black_box(
                    kdtree
                        .best_n_within::<SquaredEuclidean>(point, 0.05f64.az::<A>(), 10)
                        .min(),
                );
            });
        });
    });
}

fn bench_query_best_n_fixed_10<
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
        KdTreeFixed::<FixedU16<A>, T, K, BUCKET_SIZE, IDX>::with_capacity(initial_size);

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
                black_box(
                    kdtree
                        .best_n_within::<SquaredEuclideanFixed>(
                            point,
                            FixedU16::<A>::from_num(0.05f64),
                            10,
                        )
                        .min(),
                );
            });
        });
    });
}

criterion_group!(benches, best_10);
criterion_main!(benches);
