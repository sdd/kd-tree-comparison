use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use kiddo_v2::batch_benches;
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

use nabo::dummy_point::random_point_cloud;
use nabo::KDTree;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_10_float::<$k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("nabo {}", $subtype),
        );
    };
}

macro_rules! bench_float_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_100_float::<$k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("nabo {}", $subtype),
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
        [(f32, 2)],
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

pub fn nearest_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 100");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float_100,
        [(f32, 2)],
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

fn bench_query_nearest_10_float<'a, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    let points_to_add = random_point_cloud(initial_size as u32);

    let tree = KDTree::new_with_bucket_size(&points_to_add, BUCKET_SIZE as u32);

    let query_points = random_point_cloud(query_point_qty as u32);

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(tree.knn(10, &point));
            });
        });
    });
}

fn bench_query_nearest_100_float<'a, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    let points_to_add = random_point_cloud(initial_size as u32);

    let tree = KDTree::new_with_bucket_size(&points_to_add, BUCKET_SIZE as u32);

    let query_points = random_point_cloud(query_point_qty as u32);

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                black_box(tree.knn(100, &point));
            });
        });
    });
}

criterion_group!(benches, nearest_10, nearest_100);
criterion_main!(benches);
