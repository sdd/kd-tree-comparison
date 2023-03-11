use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration, Throughput,
};

use kiddo_v1::{distance::squared_euclidean, KdTree};

use kiddo_v2::batch_benches_parameterized;
use num_traits::Float;
use rand::distributions::{Distribution, Standard};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS_SMALL: f64 = 0.01;
const RADIUS_MEDIUM: f64 = 0.05;
const RADIUS_LARGE: f64 = 0.25;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt,  $subtype: expr) => {
        bench_query_float::<$a, $k>(
            &mut $group,
            $size,
            $radius,
            &format!("Kiddo_v1 {}", $subtype),
        );
    };
}

pub fn within_small(c: &mut Criterion) {
    within(c, RADIUS_SMALL, "small");
}

pub fn within_medium(c: &mut Criterion) {
    within(c, RADIUS_MEDIUM, "medium");
}

pub fn within_large(c: &mut Criterion) {
    within(c, RADIUS_LARGE, "large");
}

fn within(c: &mut Criterion, radius: f64, radius_name: &str) {
    let mut group = c.benchmark_group(format!("Query, within, {} radius", radius_name));
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches_parameterized!(
        group,
        bench_float,
        radius,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
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

fn bench_query_float<'a, A: Float, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) where
    f64: Cast<A>,
    Standard: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    let mut initial_points = vec![];
                    for _ in 0..size {
                        initial_points.push(rand::random::<([A; K], u32)>());
                    }
                    let mut kdtree =
                        KdTree::<A, u32, K>::with_per_node_capacity(BUCKET_SIZE).unwrap();

                    for i in 0..initial_points.len() {
                        kdtree
                            .add(&initial_points[i].0, initial_points[i].1)
                            .unwrap();
                    }

                    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
                        .into_iter()
                        .map(|_| rand::random::<[A; K]>())
                        .collect();

                    (kdtree, query_points)
                },
                |(kdtree, query_points)| {
                    black_box(query_points.iter().for_each(|point| {
                        let _res = black_box(kdtree.best_n_within_into_iter(
                            &point,
                            radius.az::<A>(),
                            10,
                            &squared_euclidean,
                        ));
                    }))
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, within_small, within_medium, within_large);
criterion_main!(benches);
