# kd-tree-comparison

This project benchmarks a few different k-d tree libraries in different scenarios.
It measures construction time as well as query time, across a matrix of tree sizes, dimensionality, and underlying data type.

Comments and contributions are welcome.

## Data Visualization

A companion webapp to this test suite exists where the results can be explored interactively, at [https://sdd.github.io/kd-tree-comparison-webapp/](https://sdd.github.io/kd-tree-comparison-webapp/).

The repository for the visualisation webapp is at [https://github.com/sdd/kd-tree-comparison-webapp](https://github.com/sdd/kd-tree-comparison-webapp)

## Libraries tested
(full disclosure: I'm the author of Kiddo)

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


## Converting the criterion NDJSON into a convenient JSON object
```bash
jq -s '.[] | select(.reason == "benchmark-complete") | with_entries(select([.key] | inside(["id", "mean"])))'  < all-benchmarks.ndjson | jq -s > all-benchmarks.json
```

## Benchmark System Details

* Processor: Ryzen 5900X (12/24 core)
* Memory: 32Gb DDR4, 3600MHz
