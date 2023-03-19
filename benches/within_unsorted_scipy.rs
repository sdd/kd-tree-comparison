use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkGroup, Criterion, PlotConfiguration,
    Throughput,
};
use criterion_polyglot::{BenchSpec, CriterionPolyglotExt};

use kiddo_v2::batch_benches;
use num_traits::Float;
use rand::distributions::{Distribution, Standard};

const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;

fn rust_float_to_py(rust_float_type_name: &str) -> String {
    format!("np.float{}", rust_float_type_name[rust_float_type_name.len()-2..].to_owned())
}

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_float::<$a, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("scipy {}", $subtype),
        );
    };
}

pub fn within(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query within radius unsorted");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float,
        [(f32, 2), (f32, 3), (f32, 4), (f64, 2), (f64, 3), (f64, 4)],
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

fn bench_query_float<A: Float, const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    group.python_benchmark(
        &*format!("{}/{}", &subtype, &initial_size),
        BenchSpec::new(&*format!(
            r#"
results = kd_tree.query_ball_point(query_pts, {}, return_sorted=False)
        "#,
            RADIUS.sqrt(),
        ))
            .with_global_init(&*format!(
                r#"
from scipy.spatial import KDTree
import numpy as np

data_pts = np.random.rand({}, {}).astype({})
query_pts = np.random.rand({}, {}).astype({})

kd_tree = KDTree(data_pts)
        "#,
                &initial_size, K, rust_float_to_py(std::any::type_name::<A>()), &query_point_qty, K, rust_float_to_py(std::any::type_name::<A>())
            )),
    );
}


criterion_group!(benches, within);
criterion_main!(benches);
