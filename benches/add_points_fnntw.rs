use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration,
};
use kiddo_v2::batch_benches;
use rand::distributions::{Distribution, Standard};

use fnntw::Tree;

const BUCKET_SIZE: usize = 32;

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$k>(&mut $group, $size, &format!("FNNTW {}", $subtype));
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
            (1_000_000, u32, u32),
            (10_000_000, u32, u32)
        ]
    );

    group.finish();
}

fn bench_add_to_empty_float<const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    Standard: Distribution<[f64; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let points_to_add: Vec<[f64; K]> = (0..size)
                        .into_iter()
                        .map(|_| rand::random::<[f64; K]>())
                        .collect();

                    points_to_add
                },
                |points_to_add| {
                    black_box({
                        let tree = Tree::new_parallel(
                            black_box(&points_to_add),
                            black_box(BUCKET_SIZE),
                            1,
                        )
                        .unwrap();
                        drop(tree);
                    })
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, add_to_empty);
criterion_main!(benches);
