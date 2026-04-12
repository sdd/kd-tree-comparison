# kd-tree-comparison

This project benchmarks a few different k-d tree libraries in different scenarios.
It measures construction time as well as query time, across a matrix of tree sizes, dimensionality, and underlying data type.

Comments and contributions are welcome.

## Data Visualization

A companion webapp to this test suite exists where the results can be explored interactively, at [https://sdd.github.io/kd-tree-comparison-webapp/](https://sdd.github.io/kd-tree-comparison-webapp/).

The repository for the visualisation webapp is at [https://github.com/sdd/kd-tree-comparison-webapp](https://github.com/sdd/kd-tree-comparison-webapp)

## Libraries tested
(full disclosure: I'm the author of Kiddo)

* [Kiddo v6.x](https://github.com/sdd/kiddo)
* [Kiddo v5.x](https://github.com/sdd/kiddo)
* [Kiddo v3.x](https://github.com/sdd/kiddo)
* [Kiddo v2.x](https://github.com/sdd/kiddo)
* [Kiddo v1.x / v0.2.x](https://github.com/sdd/kiddo_v1)
* [FNNTW](https://crates.io/crates/fnntw) v0.2.3
* [nabo-rs](https://crates.io/crates/nabo) v0.2.1
* [pykdtree](https://github.com/storpipfugl/pykdtree) v1.3.4
* [sklearn.neighbours.KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) v1.2.2
* [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) v1.10.1

## Running the benchmarks to generate NDJSON

```bash
cargo criterion --message-format json > all-benchmarks.ndjson
```

By default, the crate enables all tree/query selector features and `dims-3`, so the command above runs the full suite for 3D only.
Bench targets are now gated by three orthogonal feature families:

* `tree-*` selects the library/version variant, such as `tree-kiddo-v3-std` or `tree-scipy`
* `query-*` selects the benchmark operation family, such as `query-nearest-n` or `query-within`
* `dims-*` selects which dimensionalities are included: `dims-2`, `dims-3`, `dims-4`, or `dims-all`

For kiddo v6, the tree selector also encodes the leaf/stem layout, for example
`tree-kiddo-v6-flatvec-eytzinger`, `tree-kiddo-v6-vecofarenas-eytzinger-pf`, or
`tree-kiddo-v6-vecofarenas-donnelly-block-simd`.

To narrow the run, disable default features and combine one selector from each family, or use the aggregate helpers `all-trees` / `all-queries` / `dims-all`.

```bash
# all benchmarks for one tree variant across every supported dimension
cargo criterion --no-default-features --features tree-kiddo-v3-std,all-queries,dims-all

# one query family across every tree/library variant in 3D only
cargo criterion --no-default-features --features all-trees,query-nearest-n,dims-3

# a single tree/query combination
cargo criterion --no-default-features --features tree-nabo,query-nearest-n-within,dims-4
```

Tree size selection is also configurable through `KD_TREE_SIZE_PROFILE`. The default is `standard`, which benchmarks:

* `2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22, 2^24`

The available profiles are:

* `smoke`: `2^10, 2^14, 2^18, 2^22`
* `standard`: `2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22, 2^24`
* `extended`: `2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^21, 2^22, 2^23, 2^24, 2^25, 2^26`
* `full`: every power of two from `2^10` through `2^26`

```bash
# use the default standard size ladder
cargo criterion --no-default-features --features tree-kiddo-v3-std,query-nearest-one,dims-3

# quick cache-sensitive sweep
KD_TREE_SIZE_PROFILE=smoke cargo criterion --no-default-features --features all-trees,query-nearest-n,dims-all

# dense powers-of-two sweep for a single tree/query pairing
KD_TREE_SIZE_PROFILE=full cargo criterion --no-default-features --features tree-kiddo-v6-vecofarenas-donnelly-block-simd,query-nearest-n-within,dims-3
```


## Converting the criterion NDJSON into a convenient JSON object
```bash
jq -s '.[] | select(.reason == "benchmark-complete") | with_entries(select([.key] | inside(["id", "mean"])))'  < all-benchmarks.ndjson | jq -s > all-benchmarks.json
```

## Benchmark System Details

* Processor: Ryzen 5900X (12/24 core)
* Memory: 32Gb DDR4, 3600MHz
