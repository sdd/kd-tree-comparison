use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration,
};
use fixed::types::extra::{Unsigned, U16};
use fixed::FixedU16;
use rand::distributions::{Distribution, Standard};

use kiddo_v2::batch_benches;
use kiddo_v2::fixed::kdtree::{Axis as AxisFixed, KdTree as FixedKdTree};
use kiddo_v2::float::kdtree::{Axis, KdTree};
use kiddo_v2::test_utils::rand_data_fixed_u16_entry;
use kiddo_v2::types::{Content, Index};

const BUCKET_SIZE: usize = 32;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v2 {}", $subtype),
        );
    };
}

macro_rules! bench_empty_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_fixed_u16::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            &format!("Kiddo_v2 {}", $subtype),
        );
    };
}

pub fn add_to_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add to Empty Tree");

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
        bench_empty_float,
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

    group.finish();
}

fn bench_add_to_empty_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<([A; K], T)>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let points_to_add: Vec<([A; K], T)> = (0..size)
                        .into_iter()
                        .map(|_| rand::random::<([A; K], T)>())
                        .collect();

                    let kdtree =
                        KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(points_to_add.len());

                    (kdtree, points_to_add)
                },
                |(mut kdtree, points_to_add)| {
                    black_box(
                        points_to_add
                            .iter()
                            .for_each(|point| kdtree.add(&point.0, point.1)),
                    )
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_add_to_empty_fixed_u16<A: Unsigned, T: Content, const K: usize, IDX: Index<T = IDX>>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let mut points_to_add = Vec::with_capacity(size);
                    for _ in 0..size {
                        points_to_add.push(rand_data_fixed_u16_entry::<A, T, K>());
                    }

                    points_to_add
                },
                |points_to_add| {
                    black_box({
                        let mut kdtree =
                            FixedKdTree::<FixedU16<A>, T, K, BUCKET_SIZE, IDX>::with_capacity(size);

                        points_to_add
                            .iter()
                            .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
                    });
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, add_to_empty);
criterion_main!(benches);
