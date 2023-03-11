use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkGroup, Criterion, PlotConfiguration,
    Throughput,
};
use criterion_polyglot::{BenchSpec, CriterionPolyglotExt};

use kiddo_v2::batch_benches;
use rand::distributions::{Distribution, Standard};

const QUERY_POINTS_PER_LOOP: usize = 1_000;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_10_float::<$k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("scipy {}", $subtype),
        );
    };
}

macro_rules! bench_float_100 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_100_float::<$k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("scipy {}", $subtype),
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
        [(f64, 2), (f64, 3), (f64, 4)],
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

fn bench_query_nearest_10_float<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    group.python_benchmark(
        &*format!("{}/{}", &subtype, &initial_size),
        BenchSpec::new(
            r#"
dist, idx = kd_tree.query(query_pts, k=10)
        "#,
        )
        .with_global_init(&*format!(
            r#"
from scipy.spatial import KDTree
import numpy as np

data_pts = np.random.rand({}, {})
query_pts = np.random.rand({}, {})

kd_tree = KDTree(data_pts)
        "#,
            &initial_size, K, &query_point_qty, K
        )),
    );
}

fn bench_query_nearest_100_float<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    group.python_benchmark(
        &*format!("{}/{}", &subtype, &initial_size),
        BenchSpec::new(
            r#"
dist, idx = kd_tree.query(query_pts, k=100)
        "#,
        )
        .with_global_init(&*format!(
            r#"
from scipy.spatial import KDTree
import numpy as np

data_pts = np.random.rand({}, {})
query_pts = np.random.rand({}, {})

kd_tree = KDTree(data_pts)
        "#,
            &initial_size, K, &query_point_qty, K
        )),
    );
}

criterion_group!(benches, nearest_10, nearest_100);
criterion_main!(benches);
