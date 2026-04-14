use std::env;

pub const BENCH_SIZE_PROFILE_ENV: &str = "KD_TREE_SIZE_PROFILE";

const SMOKE_SIZES: &[usize] = &[1_024, 16_384, 262_144, 4_194_304];
const STANDARD_SIZES: &[usize] = &[
    1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216,
];
const EXTENDED_SIZES: &[usize] = &[
    1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216,
    33_554_432, 67_108_864,
];
const FULL_SIZES: &[usize] = &[
    1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576,
    2_097_152, 4_194_304, 8_388_608, 16_777_216, 33_554_432, 67_108_864,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchSizeProfile {
    Smoke,
    Standard,
    Extended,
    Full,
}

impl BenchSizeProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Standard => "standard",
            Self::Extended => "extended",
            Self::Full => "full",
        }
    }
}

pub fn parse_bench_size_profile(value: &str) -> Result<BenchSizeProfile, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "standard" | "default" => Ok(BenchSizeProfile::Standard),
        "smoke" => Ok(BenchSizeProfile::Smoke),
        "extended" => Ok(BenchSizeProfile::Extended),
        "full" => Ok(BenchSizeProfile::Full),
        _ => Err(format!(
            "Invalid {BENCH_SIZE_PROFILE_ENV} value `{value}`. Expected one of: smoke, standard, extended, full."
        )),
    }
}

pub fn bench_size_profile() -> BenchSizeProfile {
    match env::var(BENCH_SIZE_PROFILE_ENV) {
        Ok(value) => parse_bench_size_profile(&value).unwrap_or_else(|message| panic!("{message}")),
        Err(env::VarError::NotPresent) => BenchSizeProfile::Standard,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("{BENCH_SIZE_PROFILE_ENV} must be valid UTF-8")
        }
    }
}

pub fn bench_size_values() -> &'static [usize] {
    bench_size_values_for_profile(bench_size_profile())
}

pub fn bench_size_values_for_profile(profile: BenchSizeProfile) -> &'static [usize] {
    match profile {
        BenchSizeProfile::Smoke => SMOKE_SIZES,
        BenchSizeProfile::Standard => STANDARD_SIZES,
        BenchSizeProfile::Extended => EXTENDED_SIZES,
        BenchSizeProfile::Full => FULL_SIZES,
    }
}

pub fn bench_dimension_enabled(dimension: usize) -> bool {
    match dimension {
        2 => cfg!(feature = "dims-2"),
        3 => cfg!(feature = "dims-3"),
        4 => cfg!(feature = "dims-4"),
        _ => false,
    }
}

pub fn nearest_n_within_max_results(initial_size: usize) -> usize {
    match initial_size {
        ..=1_024 => 3,
        ..=4_096 => 10,
        ..=1_048_576 => 100,
        _ => 1_000,
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! kd_tree_comparison_size_t_idx {
    ( $group:ident; $callee:ident; $a:ty|$k:tt; [$(($size:literal,$t:ty,$idx:ty)),+ $(,)?] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! kd_tree_comparison_size_t_idx_parameterized {
    ( $group:ident; $callee:ident; $param:tt; $a:ty|$k:tt; [$(($size:literal,$t:ty,$idx:ty)),+ $(,)?] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, $param, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[macro_export]
macro_rules! batch_benches {
    ($group:ident, $callee:ident, [$(($a:ty, $k:tt)),+ $(,)?], $_legacy_sizes:tt ) => {{
        match $crate::bench_size_profile() {
            $crate::BenchSizeProfile::Smoke => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx!($group; $callee; $a|$k; [
                            (1_024, u16, u16),
                            (16_384, u16, u16),
                            (262_144, u32, u16),
                            (4_194_304, u32, u32)
                        ]);
                    }
                )*
            }
            $crate::BenchSizeProfile::Standard => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx!($group; $callee; $a|$k; [
                            (1_024, u16, u16),
                            (4_096, u16, u16),
                            (16_384, u16, u16),
                            (65_536, u16, u16),
                            (262_144, u32, u16),
                            (1_048_576, u32, u32),
                            (4_194_304, u32, u32),
                            (16_777_216, u32, u32)
                        ]);
                    }
                )*
            }
            $crate::BenchSizeProfile::Extended => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx!($group; $callee; $a|$k; [
                            (1_024, u16, u16),
                            (4_096, u16, u16),
                            (16_384, u16, u16),
                            (65_536, u16, u16),
                            (262_144, u32, u16),
                            (1_048_576, u32, u32),
                            (2_097_152, u32, u32),
                            (4_194_304, u32, u32),
                            (8_388_608, u32, u32),
                            (16_777_216, u32, u32),
                            (33_554_432, u32, u32),
                            (67_108_864, u32, u32)
                        ]);
                    }
                )*
            }
            $crate::BenchSizeProfile::Full => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx!($group; $callee; $a|$k; [
                            (1_024, u16, u16),
                            (2_048, u16, u16),
                            (4_096, u16, u16),
                            (8_192, u16, u16),
                            (16_384, u16, u16),
                            (32_768, u16, u16),
                            (65_536, u16, u16),
                            (131_072, u32, u16),
                            (262_144, u32, u16),
                            (524_288, u32, u16),
                            (1_048_576, u32, u32),
                            (2_097_152, u32, u32),
                            (4_194_304, u32, u32),
                            (8_388_608, u32, u32),
                            (16_777_216, u32, u32),
                            (33_554_432, u32, u32),
                            (67_108_864, u32, u32)
                        ]);
                    }
                )*
            }
        }
    }};
}

#[macro_export]
macro_rules! batch_benches_parameterized {
    ($group:ident, $callee:ident, $param:tt, [$(($a:ty, $k:tt)),+ $(,)?], $_legacy_sizes:tt ) => {{
        match $crate::bench_size_profile() {
            $crate::BenchSizeProfile::Smoke => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx_parameterized!($group; $callee; $param; $a|$k; [
                            (1_024, u16, u16),
                            (16_384, u16, u16),
                            (262_144, u32, u16),
                            (4_194_304, u32, u32)
                        ]);
                    }
                )*
            }
            $crate::BenchSizeProfile::Standard => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx_parameterized!($group; $callee; $param; $a|$k; [
                            (1_024, u16, u16),
                            (4_096, u16, u16),
                            (16_384, u16, u16),
                            (65_536, u16, u16),
                            (262_144, u32, u16),
                            (1_048_576, u32, u32),
                            (4_194_304, u32, u32),
                            (16_777_216, u32, u32)
                        ]);
                    }
                )*
            }
            $crate::BenchSizeProfile::Extended => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx_parameterized!($group; $callee; $param; $a|$k; [
                            (1_024, u16, u16),
                            (4_096, u16, u16),
                            (16_384, u16, u16),
                            (65_536, u16, u16),
                            (262_144, u32, u16),
                            (1_048_576, u32, u32),
                            (2_097_152, u32, u32),
                            (4_194_304, u32, u32),
                            (8_388_608, u32, u32),
                            (16_777_216, u32, u32),
                            (33_554_432, u32, u32),
                            (67_108_864, u32, u32)
                        ]);
                    }
                )*
            }
            $crate::BenchSizeProfile::Full => {
                $(
                    if $crate::bench_dimension_enabled($k) {
                        $crate::kd_tree_comparison_size_t_idx_parameterized!($group; $callee; $param; $a|$k; [
                            (1_024, u16, u16),
                            (2_048, u16, u16),
                            (4_096, u16, u16),
                            (8_192, u16, u16),
                            (16_384, u16, u16),
                            (32_768, u16, u16),
                            (65_536, u16, u16),
                            (131_072, u32, u16),
                            (262_144, u32, u16),
                            (524_288, u32, u16),
                            (1_048_576, u32, u32),
                            (2_097_152, u32, u32),
                            (4_194_304, u32, u32),
                            (8_388_608, u32, u32),
                            (16_777_216, u32, u32),
                            (33_554_432, u32, u32),
                            (67_108_864, u32, u32)
                        ]);
                    }
                )*
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::{
        bench_dimension_enabled, bench_size_values_for_profile, nearest_n_within_max_results,
        parse_bench_size_profile, BenchSizeProfile,
    };

    #[test]
    fn parses_profile_names() {
        assert_eq!(
            parse_bench_size_profile("smoke").unwrap(),
            BenchSizeProfile::Smoke
        );
        assert_eq!(
            parse_bench_size_profile("standard").unwrap(),
            BenchSizeProfile::Standard
        );
        assert_eq!(
            parse_bench_size_profile("extended").unwrap(),
            BenchSizeProfile::Extended
        );
        assert_eq!(
            parse_bench_size_profile("full").unwrap(),
            BenchSizeProfile::Full
        );
        assert_eq!(
            parse_bench_size_profile("default").unwrap(),
            BenchSizeProfile::Standard
        );
    }

    #[test]
    fn returns_expected_standard_sizes() {
        assert_eq!(
            bench_size_values_for_profile(BenchSizeProfile::Standard),
            &[1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216]
        );
    }

    #[test]
    fn returns_expected_full_size_range() {
        let sizes = bench_size_values_for_profile(BenchSizeProfile::Full);

        assert_eq!(sizes.first(), Some(&1_024));
        assert_eq!(sizes.last(), Some(&67_108_864));
        assert_eq!(sizes.len(), 17);
    }

    #[test]
    fn ignores_unknown_dimensions() {
        assert!(!bench_dimension_enabled(1));
        assert!(!bench_dimension_enabled(5));
    }

    #[test]
    fn nearest_n_within_caps_scale_with_size() {
        assert_eq!(nearest_n_within_max_results(1_024), 3);
        assert_eq!(nearest_n_within_max_results(4_096), 10);
        assert_eq!(nearest_n_within_max_results(1_048_576), 100);
        assert_eq!(nearest_n_within_max_results(16_777_216), 1_000);
    }
}
