mod bench_profiles;
pub use bench_profiles::{
    bench_dimension_enabled, bench_dims, bench_size_profile, bench_size_values,
    bench_size_values_for_profile, nearest_n_within_max_results, parse_bench_dims,
    parse_bench_size_profile, BenchDims, BenchSizeProfile, BENCH_DIMS_ENV, BENCH_SIZE_PROFILE_ENV,
};

#[cfg(feature = "tree-nabo")]
pub mod utils;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
