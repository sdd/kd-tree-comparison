use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, Throughput, BatchSize};
use kiddo_v2::batch_benches;
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;

use fnntw::Tree;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000_000;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$k>(&mut $group, $size, QUERY_POINTS_PER_LOOP, &format!("FNNTW {}", $subtype));
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

    group.finish();
}

fn bench_query_nearest_one_float<'a, const K: usize>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    let points_to_add: Vec<[f64; K]> = (0..initial_size)
        .into_iter()
        .map(|_| rand::random::<[f64; K]>())
        .collect();

    let tree = Tree::new(black_box(&points_to_add), BUCKET_SIZE).unwrap();

    let query_points: Vec<_> = (0..query_point_qty)
        .into_iter()
        .map(|_| rand::random::<[f64; K]>())
        .collect();

    group.bench_function(
        BenchmarkId::new(subtype, initial_size),
        |b| {
            b.iter(|| {
                for point in &query_points {
                    black_box(tree.query_nearest(&point).unwrap());
                }
            });
        },
    );
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
