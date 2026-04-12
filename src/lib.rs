mod bench_profiles;
pub use bench_profiles::{
    bench_dimension_enabled, bench_size_profile, bench_size_values, bench_size_values_for_profile,
    nearest_n_within_max_results, parse_bench_size_profile, BenchSizeProfile,
    BENCH_SIZE_PROFILE_ENV,
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
