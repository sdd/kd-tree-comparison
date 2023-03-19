use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration,
};

use kiddo_v1::KdTree;

use kiddo_v2::batch_benches;
use num_traits::Float;
use rand::distributions::{Distribution, Standard};

const BUCKET_SIZE: usize = 32;

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$a, $k>(&mut $group, $size, &format!("Kiddo_v1 {}", $subtype));
    };
}

pub fn add_to_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add to Empty Tree");

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

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

fn bench_add_to_empty_float<A: Float, const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    Standard: Distribution<([A; K], u32)>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let points_to_add: Vec<([A; K], u32)> = (0..size)
                        .into_iter()
                        .map(|_| rand::random::<([A; K], u32)>())
                        .collect();

                    let kdtree = KdTree::<A, u32, K>::with_per_node_capacity(BUCKET_SIZE).unwrap();

                    (kdtree, points_to_add)
                },
                |(mut kdtree, points_to_add)| {
                    black_box(
                        points_to_add
                            .iter()
                            .for_each(|point| kdtree.add(&point.0, point.1).unwrap()),
                    )
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, add_to_empty);
criterion_main!(benches);
