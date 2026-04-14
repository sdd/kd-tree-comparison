use std::env;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIMS: usize = 3;
const BUCKET_SIZE: usize = 32;
const DEFAULT_SIZE: usize = 65_536;
const DEFAULT_QUERIES: usize = 100;
const DEFAULT_LOOPS: usize = 4_000;
const DEFAULT_WARMUP_LOOPS: usize = 200;
const DEFAULT_RADIUS: f64 = 0.01;
const DEFAULT_SEED: u64 = 42;

#[cfg(all(
    feature = "tree-kiddo-v5-immutable",
    feature = "tree-kiddo-v6-vecofarenas-eytzinger-pf-far"
))]
compile_error!(
    "Enable exactly one of `tree-kiddo-v5-immutable` or `tree-kiddo-v6-vecofarenas-eytzinger-pf-far`."
);

#[cfg(not(any(
    feature = "tree-kiddo-v5-immutable",
    feature = "tree-kiddo-v6-vecofarenas-eytzinger-pf-far"
)))]
compile_error!(
    "Enable one of `tree-kiddo-v5-immutable` or `tree-kiddo-v6-vecofarenas-eytzinger-pf-far` to build `profile_nearest_n_within`."
);

#[derive(Clone, Copy, Debug)]
struct Config {
    size: usize,
    queries: usize,
    loops: usize,
    warmup_loops: usize,
    radius: f64,
    max_results: Option<usize>,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            size: DEFAULT_SIZE,
            queries: DEFAULT_QUERIES,
            loops: DEFAULT_LOOPS,
            warmup_loops: DEFAULT_WARMUP_LOOPS,
            radius: DEFAULT_RADIUS,
            max_results: None,
            seed: DEFAULT_SEED,
        }
    }
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);

        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("Missing value for `{flag}`"))?;

            match flag.as_str() {
                "--size" => config.size = parse_usize(&flag, &value)?,
                "--queries" => config.queries = parse_usize(&flag, &value)?,
                "--loops" => config.loops = parse_usize(&flag, &value)?,
                "--warmup-loops" => config.warmup_loops = parse_usize(&flag, &value)?,
                "--radius" => config.radius = parse_f64(&flag, &value)?,
                "--max-results" => {
                    config.max_results = match value.as_str() {
                        "auto" => None,
                        _ => Some(parse_usize(&flag, &value)?),
                    };
                }
                "--seed" => config.seed = parse_u64(&flag, &value)?,
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("Unknown flag `{flag}`")),
            }
        }

        if config.size == 0 {
            return Err("`--size` must be greater than zero".to_string());
        }
        if config.queries == 0 {
            return Err("`--queries` must be greater than zero".to_string());
        }
        if config.loops == 0 {
            return Err("`--loops` must be greater than zero".to_string());
        }
        if let Some(max_results) = config.max_results {
            if max_results == 0 {
                return Err("`--max-results` must be greater than zero or `auto`".to_string());
            }
        }

        Ok(config)
    }
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("Invalid value `{value}` for `{flag}`: {err}"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("Invalid value `{value}` for `{flag}`: {err}"))
}

fn parse_f64(flag: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|err| format!("Invalid value `{value}` for `{flag}`: {err}"))
}

fn print_usage() {
    eprintln!(
        "Usage: profile_nearest_n_within [--size N] [--queries N] [--loops N] [--warmup-loops N] [--radius F64] [--max-results auto|N] [--seed N]"
    );
}

fn effective_max_results(config: &Config) -> NonZeroUsize {
    NonZeroUsize::new(
        config
            .max_results
            .unwrap_or_else(|| kd_tree_comparison::nearest_n_within_max_results(config.size)),
    )
    .unwrap()
}

fn mix_results(mut checksum: u64, len: usize, first: Option<(f64, u32)>, last: Option<(f64, u32)>) -> u64 {
    checksum = checksum
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(len as u64);

    if let Some((distance, item)) = first {
        checksum ^= distance.to_bits().rotate_left(7);
        checksum = checksum.wrapping_add(item as u64);
    }

    if let Some((distance, item)) = last {
        checksum ^= distance.to_bits().rotate_left(29);
        checksum = checksum.wrapping_add((item as u64).rotate_left(13));
    }

    checksum
}

#[cfg(feature = "tree-kiddo-v5-immutable")]
mod impls {
    use super::{black_box, mix_results, Config, DIMS};
    use kiddo_v5::float::distance::SquaredEuclidean;
    use kiddo_v5::immutable::float::kdtree::ImmutableKdTree;
    use rand::rngs::StdRng;
    use rand::Rng;

    pub const IMPL_NAME: &str = "kiddo_v5_immutable";

    pub type Tree = ImmutableKdTree<f64, u32, DIMS, { super::BUCKET_SIZE }>;

    pub fn build_tree(config: &Config, rng: &mut StdRng) -> Tree {
        let points: Vec<[f64; DIMS]> = (0..config.size).map(|_| rng.gen()).collect();
        ImmutableKdTree::new_from_slice(&points)
    }

    pub fn run_batch(
        tree: &Tree,
        query_points: &[[f64; DIMS]],
        radius: f64,
        max_results: std::num::NonZeroUsize,
    ) -> u64 {
        let mut checksum = 0u64;

        for point in query_points {
            let results = black_box(tree.nearest_n_within::<SquaredEuclidean>(
                black_box(point),
                black_box(radius),
                max_results,
                true,
            ));

            let first = results.first().map(|nn| (nn.distance, nn.item));
            let last = results.last().map(|nn| (nn.distance, nn.item));
            checksum = mix_results(checksum, results.len(), first, last);
        }

        checksum
    }
}

#[cfg(feature = "tree-kiddo-v6-vecofarenas-eytzinger-pf-far")]
mod impls {
    use super::{black_box, mix_results, Config, DIMS};
    use kiddo_v6::dist::DistanceMetricCore;
    use kiddo_v6::kd_tree::leaf_strategies::VecOfArenas;
    use kiddo_v6::kd_tree::KdTree;
    use kiddo_v6::stem_strategies::eytzinger_pf_far::EytzingerPfFar;
    use kiddo_v6::SquaredEuclidean;
    use rand::rngs::StdRng;
    use rand::Rng;

    pub const IMPL_NAME: &str = "kiddo_v6_vecofarenas_eytzinger_pf_far";

    pub type Tree = KdTree<
        f64,
        u32,
        EytzingerPfFar<DIMS, 8>,
        VecOfArenas<f64, u32, DIMS, { super::BUCKET_SIZE }>,
        DIMS,
        { super::BUCKET_SIZE },
    >;

    pub fn build_tree(config: &Config, rng: &mut StdRng) -> Tree {
        let points: Vec<[f64; DIMS]> = (0..config.size).map(|_| rng.gen()).collect();
        KdTree::new_from_slice(&points)
    }

    pub fn run_batch(
        tree: &Tree,
        query_points: &[[f64; DIMS]],
        radius: f64,
        max_results: std::num::NonZeroUsize,
    ) -> u64 {
        let widened_radius = <SquaredEuclidean<f64> as DistanceMetricCore<f64>>::widen_coord(radius);
        let mut checksum = 0u64;

        for point in query_points {
            let results = black_box(tree.nearest_n_within::<SquaredEuclidean<f64>>(
                black_box(point),
                black_box(widened_radius),
                max_results,
                true,
            ));

            let first = results.first().map(|nn| (nn.distance, nn.item));
            let last = results.last().map(|nn| (nn.distance, nn.item));
            checksum = mix_results(checksum, results.len(), first, last);
        }

        checksum
    }
}

fn main() {
    let config = Config::from_args().unwrap_or_else(|message| {
        eprintln!("{message}");
        print_usage();
        std::process::exit(2);
    });

    let mut point_rng = StdRng::seed_from_u64(config.seed);
    let mut query_rng = StdRng::seed_from_u64(config.seed.wrapping_add(1));

    let tree = impls::build_tree(&config, &mut point_rng);
    let query_points: Vec<[f64; DIMS]> = (0..config.queries).map(|_| query_rng.gen()).collect();
    let max_results = effective_max_results(&config);

    let mut warmup_checksum = 0u64;
    for _ in 0..config.warmup_loops {
        warmup_checksum ^= impls::run_batch(&tree, &query_points, config.radius, max_results);
    }

    let start = Instant::now();
    let mut checksum = 0u64;
    for _ in 0..config.loops {
        checksum ^= impls::run_batch(&tree, &query_points, config.radius, max_results);
    }
    let elapsed = start.elapsed();

    println!(
        "{{\"impl\":\"{}\",\"size\":{},\"queries\":{},\"loops\":{},\"warmup_loops\":{},\"radius\":{},\"max_results\":{},\"seed\":{},\"warmup_checksum\":{},\"checksum\":{},\"elapsed_ns\":{}}}",
        impls::IMPL_NAME,
        config.size,
        config.queries,
        config.loops,
        config.warmup_loops,
        config.radius,
        max_results.get(),
        config.seed,
        warmup_checksum,
        checksum,
        elapsed.as_nanos(),
    );
}
