## Running the benchmarks to generate NDJSON

```bash
cargo criterion --message-format json > all-benchmarks.ndjson
```


## Converting the criterion NDJSON into a convenient JSON object
```bash
jq -s '.[] | select(.reason == "benchmark-complete") | with_entries(select([.key] | inside(["id", "mean"])))'  < all-benchmarks.ndjson | jq -s > all-benchmarks.json
```
