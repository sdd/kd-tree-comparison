use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkGroup,
    Criterion, PlotConfiguration,
};
use criterion_polyglot::{BenchSpec, CriterionPolyglotExt};

use kiddo_v2::batch_benches;
use rand::distributions::{Distribution, Standard};

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$k>(&mut $group, $size, &format!("pykdtree {}", $subtype));
    };
}

pub fn add_to_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add to Empty Tree");

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_empty_float,
        [(f64, 2), (f64, 3), (f64, 4)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32)
        ]
    );

    /*batch_benches!(
        group,
        bench_empty_float,
        [(f64, 3)],
        [
            (100_000, u32, u16)
        ]
    );*/

    group.finish();
}

fn bench_add_to_empty_float<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    group.python_benchmark(
        &*format!("{}/{}", &subtype, &qty_to_add),
        BenchSpec::new(r#"
kd_tree = KDTree(data_pts)
        "#)
        .with_global_init(&*format!(r#"
from pykdtree.kdtree import KDTree
import numpy as np
data_pts = np.random.rand({}, {})
        "#, &qty_to_add, K))
    );
}

criterion_group!(benches, add_to_empty);
criterion_main!(benches);
