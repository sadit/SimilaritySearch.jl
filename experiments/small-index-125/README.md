# The small-index scan of `search` (issue #125)

Below `SMALLINDEX` items, `search(bs::BeamSearch, ...)` scans the whole database instead of
running the beam. Up to 1.6.5 that scan ignored `vstate`, which broke `allknn`, `rebuild` and
`neardup`. The fix sends the scan through `enqueue_item!`. This directory measures which threshold
to keep: 1 (no scan), 8, 16, 32 or 64 (the shipped value).

- `threshold.jl`: random subsets of SISAP 2025 `ccnews` train, 1000 `itest` queries, gold by
  `ExhaustiveSearch` on each subset, `NormCosine`, k = min(10, n). Five subsets per size (two at
  100,000). Two builds: sequential (`parallel_block=1`) and parallel (64 threads, so
  `parallel_block = 512`). Two searches: the beam the autotuning leaves, and a fixed
  `BeamSearch(bsize=8, Δ=1.0)`.
- `summarize.jl`: means over subsets, written to `results/summary.md`.
- `results/`: the raw rows, the log and the summary.

## What was found

Fixed beam, recall@k (mean distance evaluations per query):

| n | build | T=1 | T=8 | T=16 | T=32 | T=64 |
|---|---|---|---|---|---|---|
| 16 | seq | 0.857 (14) | 0.934 (15) | 0.984 (16) | 1.000 (16) | 1.000 (16) |
| 32 | seq | 0.974 (30) | 0.980 (30) | 0.989 (31) | 0.991 (31) | 1.000 (32) |
| 63 | seq | 0.974 (53) | 0.976 (54) | 0.979 (54) | 0.980 (54) | 1.000 (63) |
| 63 | par | 0.983 (55) | 0.983 (55) | 0.983 (55) | 0.983 (55) | 1.000 (63) |
| 128 | seq | 0.961 (84) | 0.961 (84) | 0.961 (85) | 0.961 (85) | 0.962 (85) |
| 1,024 | seq | 0.930 (235) | 0.929 (235) | 0.931 (236) | 0.932 (236) | 0.928 (234) |
| 10,000 | seq | 0.906 (364) | 0.906 (364) | 0.905 (363) | 0.906 (364) | 0.906 (365) |
| 100,000 | seq | 0.884 (481) | 0.888 (480) | 0.889 (479) | 0.885 (476) | 0.886 (478) |
| 100,000 | par | 0.888 (480) | 0.887 (476) | 0.882 (476) | 0.886 (477) | 0.886 (477) |

- From 128 items on, the threshold does not change the graph: the five columns agree within
  0.006. The scan decides at most the first 63 neighborhoods of a sequential build. A parallel
  build with 64 threads never uses it, because its first block of 512 items is already compared
  exhaustively among itself.
- The autotuned columns (in `results/summary.md`) move by up to 0.02 between runs at 1,024 items
  and more, with no order among thresholds. That spread comes from the stochastic tuning.
- Below the threshold the answer is exact. At 63 items the scan costs 63 evaluations, and the beam
  costs 47-55 for a recall of 0.94-0.98.
- Without the scan (T=1), a sequential build of 16 items reaches only 0.857.

The threshold stays at 64: it costs nothing on the large indexes `SearchGraph` is built for,
and an index below 64 items answers exactly.
