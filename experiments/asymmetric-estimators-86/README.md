# Asymmetric estimators on ccnews (issue #86)

The two scripts behind the tables posted on
[issue #86](https://github.com/sadit/SimilaritySearch.jl/issues/86): an `AsymmetricSearchGraph`
over the SISAP 2025 `ccnews` benchmark (603,664 x 384, the first 1000 `itest` queries,
recall@10 against the file's gold), one row per estimator, with the estimator's own exhaustive
ceiling next to the graph.

- `rabitq-ccnews.jl`: `RaBitQ.RaBitQCosine` (sign bits + three scalars) and `RaBitQRefined` with
  `RaBitQExactFallback` (`Float32`/`Float16`) or `RaBitQVectorFallback(SQgu4, ...)` at several `τ`.
- `sqencoder-ccnews.jl`: `ScalarQuant.SQEncoder` for the six quantizer modules, with a QR
  rotation and with `nothing`.

Both expect `~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5` and an environment with
`HDF5` and this checkout of `SimilaritySearch` (`Pkg.develop(path=".")`). Run with `-t auto`:
the builds are parallel, the per-query timings single-threaded. The `own tuning` and `build s`
columns vary run to run with the stochastic tuning callback; compare the fixed-beam columns,
`b=8` and `b=32` (`BeamSearch(; bsize=b, Δ=1.0)` set on the same graph).
