use std::fs;
use std::fs::File;
use std::io::Write;
use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration,
};
use criterion::SamplingMode::Flat;
use rand::distributions::{Distribution, Standard};

use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Deserialize, Infallible};

use kiddo_v2::KdTree;
use kiddo_v2::test_utils::rand_data_fixed_u16_entry;
use kiddo_v2::types::{Content, Index};

const BUCKET_SIZE: usize = 32;

const BUFFER_LEN: usize = 300_000_000;
const SCRATCH_LEN: usize = 300_000_000;

pub fn serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Serialize to file");

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    group.sample_size(10);
    group.sampling_mode(Flat);

    for size in [100_000, 1_000_000, 10_000_000] {
        bench_serialize_f64_3d(&mut group, size, "f64 3D");
    }

    group.finish();
}

fn bench_serialize_f64_3d(
    group: &mut BenchmarkGroup<WallTime>,
    size: usize,
    subtype: &str,
) {
    let points_to_add: Vec<([f64; 3], usize)> = (0..size)
        .into_iter()
        .map(|_| rand::random::<([f64; 3], usize)>())
        .collect();

    let mut kdtree: KdTree::<f64, 3> = KdTree::with_capacity(size);
    points_to_add.iter().for_each(|(point, idx)| {
        kdtree.add(point, *idx);
    });

    group.bench_with_input(
        BenchmarkId::new(subtype, size),
        &size,
        |b, &size| {
            b.iter_batched(
                || {},
                |_| {
                    black_box({
                        let mut file = File::create("/tmp/kiddo-tree.rkyv").expect("Could not create temp file '/tmp/kiddo-tree.rkyv'");
                        serialize_to_rkyv_f64_3d(&mut file, &kdtree);
                    });
                },
                BatchSize::LargeInput
            );
        },
    );

    fs::remove_file("/tmp/kiddo-tree.rkyv").expect("Could not delete temp file '/tmp/kiddo-tree.rkyv'");
}

fn serialize_to_rkyv_f64_3d(file: &mut File, tree: &KdTree<f64, 3>)
{
    let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);
    unsafe {
        serialize_scratch.set_len(SCRATCH_LEN);
    }
    serialize_buffer.clear();
    let mut serializer = CompositeSerializer::new(
        AlignedSerializer::new(&mut serialize_buffer),
        BufferScratch::new(&mut serialize_scratch),
        Infallible,
    );
    serializer
        .serialize_value(tree)
        .expect("Could not serialize with rkyv");
    let buf = serializer.into_serializer().into_inner();
    file.write(&buf)
        .expect("Could not write serialized rkyv to file");
}

criterion_group!(benches, serialize);
criterion_main!(benches);
