use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration,
};
use rand::distributions::{Distribution, Standard};

use kiddo_next::float::kdtree::Axis;
use kiddo_next::float_leaf_slice::leaf_slice::LeafSliceFloat;
use kiddo_next::immutable::float::kdtree::ImmutableKdTree;
use kiddo_next::types::Content;
use kiddo_v3::batch_benches;

const BUCKET_SIZE: usize = 32;

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$a, $t, $k>(
            &mut $group,
            $size,
            &format!("Kiddo_v5_immutable_dynamic {}", $subtype),
        );
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

    batch_benches!(
        group,
        bench_empty_float,
        [(f32, 2), (f32, 3), (f32, 4)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16)
        ]
    );

    group.finish();
}

fn bench_add_to_empty_float<A: Axis, T: Content, const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    Standard: Distribution<[A; K]>,
    usize: az::Cast<T>,
    A: Axis + LeafSliceFloat<T, K> + 'static,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let points_to_add: Vec<[A; K]> = (0..size)
                        .into_iter()
                        .map(|_| rand::random::<[A; K]>())
                        .collect();

                    points_to_add
                },
                |points_to_add| {
                    black_box({
                        ImmutableKdTree::<A, T, K, BUCKET_SIZE>::new_from_slice(
                            &points_to_add,
                        );
                    })
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, add_to_empty);
criterion_main!(benches);
