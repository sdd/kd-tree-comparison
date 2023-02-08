use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration, Throughput,
};

use kiddo_v1::{KdTree, distance::squared_euclidean};

use kiddo_v2::batch_benches;
use num_traits::Float;
use rand::distributions::{Distribution, Standard};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_10_float::<$a, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("Kiddo_v1 {}", $subtype)
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


fn bench_query_nearest_10_float<A: Float, const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
Standard: Distribution<([A; K], u32)>,
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
                    let mut kdtree = KdTree::<A, u32, K>::with_per_node_capacity(
                        BUCKET_SIZE
                    ).unwrap();

                    for i in 0..initial_points.len() {
                        kdtree.add(&initial_points[i].0, initial_points[i].1).unwrap();
                    }

                    let query_points: Vec<_> = (0..query_point_qty)
                    .into_iter()
                    .map(|_| rand::random::<[A; K]>())
                    .collect();

                    (kdtree, query_points)
                },
                |(kdtree, query_points)| {
                    black_box(
                        query_points
                            .iter()
                            .for_each(|point| {
                                let _res = black_box(kdtree.nearest(&point, 10, &squared_euclidean).unwrap());
                            }),
                    )
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, nearest_10);
criterion_main!(benches);
