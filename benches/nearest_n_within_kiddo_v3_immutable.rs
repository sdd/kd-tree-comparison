use std::collections::HashMap;
use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rand::distributions::{Distribution, Standard};

use kiddo_next::batch_benches_parameterized;
use kiddo_next::float::distance::SquaredEuclidean;
use kiddo_next::float::kdtree::Axis;
use kiddo_next::float_leaf_simd::leaf_node::BestFromDists;
use kiddo_next::immutable::float::kdtree::ImmutableKdTree;
use kiddo_next::types::Content;

use rayon::prelude::*;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;
const RADIUS: f64 = 0.01;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $radius:tt,  $subtype: expr) => {
        bench_query_float::<$a, $t, $k>(
            &mut $group,
            $size,
            $radius,
            &format!("Kiddo_v3_immutable {}", $subtype),
        );
    };
}

fn within(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query nearest n within radius");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches_parameterized!(
        group,
        bench_float,
        RADIUS,
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

    batch_benches_parameterized!(
        group,
        bench_float,
        RADIUS,
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

fn bench_query_float<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    radius: f64,
    subtype: &str,
) where
    A: BestFromDists<T, 32>,
    usize: Cast<T>,
    f64: Cast<A>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    let initial_points: Vec<_> = (0..initial_size)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let kdtree = ImmutableKdTree::<A, T, K, BUCKET_SIZE>::new_from_slice(&initial_points);

    let query_points: Vec<_> = (0..QUERY_POINTS_PER_LOOP)
        .into_iter()
        .map(|_| rand::random::<[A; K]>())
        .collect();

    let max_results_map =  HashMap::from([
        (100usize, 3usize),
        (1_000, 10),
        (10_000, 100),
        (100_000, 100),
        (1_000_000, 100),
        (10_000_000, 1000),
    ]);

    group.bench_function(BenchmarkId::new(subtype, initial_size), |b| {
        b.iter(|| {
            query_points.par_iter().for_each(|point| {
                let max_results = *max_results_map.get(&initial_size).unwrap();

                black_box(
                    kdtree.nearest_n_within::<SquaredEuclidean>(point, radius.az::<A>(), max_results, true)
                );
            });
        });
    });
}

criterion_group!(benches, within);
criterion_main!(benches);
