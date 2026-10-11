[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/SimilaritySearch.jl/dev)
[![Build Status](https://github.com/sadit/SimilaritySearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/SimilaritySearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/SimilaritySearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/SimilaritySearch.jl)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04442/status.svg)](https://doi.org/10.21105/joss.04442)

# SimilaritySearch.jl

SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for `SearchGraph,` a fast and flexible search index using any metric function. It is designed to support multithreading in most of its functions and structures.

The package provides the following indexes:

- `ParallelExhaustiveSearch`: A brute force search index where each query is solved using all available threads.
- `ExhaustiveSearch`: A brute force search index, each query is solved using a single thread.
- `SearchGraph`: An approximate search index with parallel construction.
- `InvertedFiles.InvertedFile`: An inverted-index structure for sparse vectors, Maximum Inner Product Search (MIPS), and set search (Jaccard, Dice, Intersection, CosineSet, RogersTanimoto, and any other distance via a direct-evaluate fallback).

The main set of functions are:

- `search`: Solves a single query.
- `searchbatch`: Solves a set of queries.
- `allknn`: Computes the $k$ nearest neighbors for all elements in an index.
- `neardup`: Removes near-duplicates from a metric dataset.
- `closestpair` / `closestpairs`: Computes the closest pair (or the $k$ closest pairs) in a metric dataset.
- `bichromatic_closestpair` / `bichromatic_kclosestpairs` / `bichromatic_metricjoin`: The same closest-pair family, but **between two different datasets** (e.g., "for every object in `B`, what's its closest match in `A`?"), plus a metric join when neither a fixed radius nor a match count per element is known ahead of time.
- `fft` / `dnet` / `randsel` / `multirandsel`: Pick a diverse, well-separated, or representative subset of a dataset (the `KCenters` submodule).

The precise definitions of these functions and the complete set of functions and structures can be found in the [documentation](https://sadit.github.io/SimilaritySearch.jl/dev), which also includes a from-scratch [tutorial series](https://sadit.github.io/SimilaritySearch.jl/dev/tutorial/) covering databases, distances, `SearchGraph`, these whole-dataset operations, parallelism, persistence, logging, inverted files, and quantization/bit sketches.

# What each release series brings

The package follows semantic versioning; a series (`1.6.x`, `1.5.x`, ...) adds features without
removing any that worked before. Patch releases inside a series are fixes and performance work.
1.6 is the one exception, and it is stated below.

## 1.6

### Migrating from 1.5: three goals were removed

This series removes three optimization goals. A minor release does not normally remove anything,
so the exception is stated here instead of left for a reader to find.

- `ParetoRecall(r)` and `ParetoRadius(r)` are replaced by **`MinRecall(r)`**. Neither computed a
  Pareto front. Both were a weighted sum of squares, and the cost term was normalized by the
  maximum of the initial population, so the trade-off they selected changed with that population.
  `MinRecall` minimizes `goalvalue`: the log cost plus a smooth hinge on the target. Its
  `tradeoff` keyword states the cost factor you accept per 1% of quality near the target. Use it
  to say what the Pareto goals were trying to say.
- `OptRadius(tol)` is replaced by **`MaxMatchError(; maxerror)`**. `OptRadius` targeted a covering
  radius within a tolerance, and you could not choose that tolerance without looking at the
  distances first. `MaxMatchError` keeps the idea and reads the scale from each query's own
  neighborhood, so `maxerror` is a fraction of that neighborhood's spread and carries from one
  dataset to another. The tutorial section *`MaxMatchError`: A Distance-Based Alternative to
  `MinRecall`* shows how to calibrate one against a `MinRecall` target you already know.

### What is new

- **The goals minimize a smooth objective.** `MinRecall(t)` and `MaxMatchError(e)` used to rank any configuration
  below the target behind any configuration above it, whatever the costs: blind to a configuration a hair
  short at a fraction of the cost, and a coin toss within the noise of the recall estimate (0.04 with 64
  tuning queries). They now minimize `goalvalue`, the log cost plus a finite-support hinge on the target,
  with three knobs: `tradeoff`, the cost factor accepted per 1% of quality near the target (default `1.5`);
  `width`, the hinge's half-width, derived by default from the quality's standard error over the tuning
  queries; and `transition`, the hinge's zone as multipliers of `width` (`(-1, 1)`, the default, lands within
  a width above the target; `(0, 2)` lands like a hard threshold; `(-2, 0)` treats the target as a floor;
  any other pair, asymmetric included, works). Measured
  on SISAP 2025 `ccnews` against the hard threshold and a `softplus` hinge: the centered hinge lands at
  0.900-0.913 for a target of 0.9 and barely moves with `tradeoff`, `softplus` overshoots by two to three
  widths, and the cost is in nats, so the normalization by the initial population's maximum cost is gone.
- **The queries that tune an index are declared, not inferred.** An internal query -- an object the
  index already stores -- is a vertex of the graph: a search reaches it at distance 0 and reads its
  adjacency list in one step, and that list is close to the answer. An external query has to reach
  its neighbors through ordinary links. Tuning with internal queries without accounting for this
  picks parameters for an easier problem: measured on two SISAP 2025 benchmarks, `bsize` and `Δ`
  came out at the cheap end of their ranges and recall@10 against real queries fell from 0.90 to
  0.69. Internal queries are now masked, from the search and from the gold alike. Which ones are
  internal is said by `queries_identifiers`, beside `queries`: the first says what to search with,
  the second says where those objects are stored. A query is masked if and only if its identifier
  is given; the previous rule, which read it from the container's type, is gone, and
  `optimize_index!` warns on the one shape that rule used to cover. Giving both is what a quantized
  index needs, since its database holds codes and an identifier alone cannot produce a raw query
  object. `numqueries` now means how many queries one optimization uses, whatever the source: a
  draw from a pool of identifiers when one is named, a sample of the index when not. Insertions set
  a pool aside once (`TUNINGPOOLSIZE`) so that the optimizations the construction callback runs are
  scored on the same population, and so that `SearchGraph` and `AsymmetricSearchGraph` tune the same
  way, which matters whenever the two are compared. See the tutorial section
  *Choosing the queries that tune an index*.
- **A tuning query whose own cluster empties its gold is dropped.** With `k=10`, an object in a
  near-duplicate cluster of 11 or more has nothing left after its cluster is masked, and
  `recallscore` divides by the gold's size. The `NaN` reached the mean, every comparison against it
  was false, and the solver returned an arbitrary configuration without raising. On SISAP 2025
  `ccnews`, 7.9% of the objects are such a case, so 99.5% of tuning runs drew at least one; the
  spread of a rebuilt cell grew 4.6x for float32 and 10.5x for a quantized one against the same
  configurations without folding. `MaxMatchError` failed the other way, scoring those queries as a
  perfect match, which preserved orderings but scaled its mean. Both goals drop them, so the two
  still tune on the same queries. When every query is degenerate the masking is given up with a
  warning instead, since raising would take `index!` down with it.
- **`neardup` is a numerical zero, and negative radii are rejected.** Two bit-identical vectors
  usually do not evaluate to `0f0`: on SISAP 2025 `ccnews` and `yahooaq` half of such pairs do, a
  sixth come out negative, and the error never exceeds 6 ulps, so a literal radius of zero missed
  about a third of the exact duplicates. A non-negative `neardup` is raised to
  `NEARDUP_NUMERICAL_ZERO` (`1f-5`), an order of magnitude above that error and three to four orders
  below any real distance on those datasets; `typemin(Float32)` stays exactly as it is, since
  raising it would turn a mechanism that never fires into one that fires on every single-entry
  neighborhood. A negative radius is no longer accepted: the distances that evaluate below zero are
  the ones wrapped to find farthest objects, and under those identical objects are the farthest of
  all, so folding near duplicates there folds what by construction never resembles anything.
  `Selection.neardup`'s own `ϵ` takes the same floor.
- **Near duplicates fold into members.** `Neighborhood(neardup=ϵ)` makes an object whose nearest indexed object
  lies within `ϵ` a member of that object's cluster instead of a node: one edge to the representative, nothing
  linking to it, never visited. `search` answers with representatives, one per cluster, and the second stage,
  `expand`/`expand!`, gives the raw neighbors back on any result form, each member with its evaluated distance.
  Members count in `length`, `optimize_index!` expands before scoring and masks a query's whole cluster,
  `rebuild` keeps them. On SISAP 2025 `ccnews`, 27% exact duplicates: 30% fewer edges, the build 37% faster,
  recall up on every query and from 0.72 to 0.86 on the queries with ten copies in the database. Off by default.
  `SearchGraph` gained the field `members` for it: a graph stored before 1.6 and read back field by field
  is rebuilt with `SearchGraph(dist, db, adj, hints, algo, len)`, which fills the field with an empty
  `Members`, the only value a pre-1.6 graph can hold. `neardup` is validated: `typemin(Float32)` (off) or a
  finite non-negative distance, raised to the numerical zero described above.
- **`matcherror` is bounded, and its parameters say what they are.** A position never costs more than
  `maxdeviation` spreads, which is also what a missing position costs, so the per-query score lies in
  `[0, maxdeviation ^ exponent]` and its mean over queries means something: on SISAP 2025 `ccnews`, without
  the cap, ten queries whose gold neighbors were exact duplicates made 85% of the mean over 10,500 queries
  and `MaxMatchError` tuned to the same configuration for any target. `p`, `η` and `minspread` are now
  `exponent`, `maxdeviation` and `spreadfloor`, as keywords of `matcherror`, `macromatcherror` and
  `MaxMatchError`.
- **`ParetoRecall`, `ParetoRadius` and `OptRadius` are gone.** The first two were not Pareto fronts but a
  sum of squares with the cost normalized by the initial population's maximum, so the trade-off they picked
  depended on that population, and a trade-off chosen at construction did not carry over to the search: a
  better graph is both more accurate and faster at a fixed beam. `OptRadius` targeted a covering radius within
  a tolerance, which could not be set without a prior look at the distances; `MaxMatchError` is the same idea
  with the scale read off each query's own neighborhood. `MinRecall` and `MaxMatchError` remain; a
  bi-objective goal will return as a smooth, explicitly weighted combination.

### 1.6.1

- **The tuning pool no longer starves the construction callbacks of a large insertion.** The pool
  is drawn once over the whole range being inserted, and 1.6.0 dropped the identifiers not yet
  inserted from each callback's optimization; on a 600K build the early callbacks were left with
  one or two queries, every configuration failed and the previous parameters stayed (#107). An
  identifier the index has not reached now counts as an external query for that call, masked only
  once it is a vertex. Identifiers beyond the database raise `ArgumentError`.

### 1.6.2

- **Prepared queries for the mixed distances** (#110, #111). `encodequery(::SQEncoder, q)` returns an
  `SQQuery`: the rotated query with its sums and an integer image on its own range (15 bits against
  8-bit codes, 8 bits against 4- and 2-bit codes). `SqL2`, `NormCosine` and `Cosine` against it are the
  expansion over the stored code sums plus one integer dot product, so a query against codes costs
  47, 45 and 42 ns per pair at 2, 4 and 8 bits on a Xeon Silver 4216 at 384 dimensions, against 84,
  74 and 48 before (code against code: 31, 27, 28). It is still an `AbstractVector{Float32}`, so plain
  query paths keep working; accuracy against the `Float32` query is within 6e-5 at 8 bits and 0.3-0.5%
  at 4 and 2, under the codes' own error.

### 1.6.3

- **The beam search expands a neighbourhood in two passes and prefetches its children** (#113). The
  first pass over the neighbours of a popped vertex only reads the visited set and asks the hardware
  for the storage of every child that will be evaluated; the second pass evaluates them as before, so
  the cache misses of a whole neighbourhood overlap instead of being paid one after another. Results
  are identical. `prefetch_item(db, i)` is the hook, with methods for `MatrixDatabase`,
  `BlockMatrixDatabase`, `MMapMatrixDatabase`, `SubDatabase`, `VectorDatabase` of vectors or strings,
  and `QuantDatabase` (codes plus the stored sums and the per-vector quantizer); `prefetchable(db)`
  says whether a database has one, and the search skips the pass otherwise. Items of up to 512 bytes
  are prefetched whole into every cache level, items up to 1024 bytes get their four leading lines
  into L2 (the hardware streamer follows), larger items are left alone: Float32 vectors at 384
  dimensions lost 4-13% with either policy, since that search is half arithmetic and the pass costs
  more than the misses it hides. Measured at 64 threads on three fresh builds per variant against
  three or more controls, static adjacency, queries per second at equal recall: 8-bit codes on ccnews
  1.06-1.10× at recall 0.90, 1.20-1.27× at 0.95, 1.34× on external queries (one thread: 1.15× and
  1.36×); per-vector 8-bit codes on yahooaq 1.17×, 1.21×, 1.23×; Float16 vectors 1.12× and 1.18×;
  clustered sets under Jaccard 1.02-1.08×; byte strings under Levenshtein 1.15× (1.23× on one
  thread); Float32 vectors unchanged. Construction time does not change.

### 1.6.4

- **The per-vector scalar quantizer places each vector's range by a policy, chosen by width** (#116, #117).
  Up to 1.6.3 `SQu2`, `SQu4` and `SQu8` mapped each vector's extrema onto the codes, so one outlying
  coordinate coarsened all the others; with three or fifteen levels that left the bulk of a vector
  on one or two codes. `quantvector!`, `SQVec{B}(v)`, `SQEncoder` and the per-vector databases now
  take `range=`, a `RangePolicy` (all in `SimilaritySearch.ScalarQuant`, not re-exported):
  `ExtremaRange()` (the old rule), `FixedRange(k)` (`mean ± k·σ`), `CalibratedRange()` (one `k`
  fitted on a sample of the data, then only the vector's mean and σ: ~1 µs per vector at 384
  dimensions), `HistogramRange(bins=64)` (`k` searched per vector on a one-pass histogram of the
  deviations, ~3 µs), `RefinedRange(inner)` (the real distortion at `inner`'s `k` and its two
  neighbours) and `ExactRange()` (the full distortion search, ~32 µs). The default `AutoRange()`
  resolves to `CalibratedRange()` at 2 bits, `HistogramRange()` at 4 and `ExtremaRange()` at 8. An
  `SQEncoder` built from data calibrates and keeps its policy; a per-vector database built from a
  matrix only resolves it, so it holds the codes growing it with `push_item!` would; an encoder
  stored before 1.6.4 keeps producing extrema. Measured offline on yahooaq and ccnews (100K
  vectors, 384 dimensions, exhaustive recall@10, symmetric / asymmetric kernels): at 2 bits the
  extrema gave 0.544 / 0.616 and 0.569 / 0.621, the calibrated `k` 0.741 / 0.803 and 0.742 / 0.802;
  at 4 bits 0.913 / 0.931 and 0.900 / 0.917 against the histogram's 0.919 / 0.933 and 0.915 / 0.926;
  at 8 bits the extrema were the only policy that did not lose (0.994 / 0.995, 0.981 / 0.991). The
  histogram with 32 or 16 bins was below 64 at every width for 0.1-0.2 µs less.

### 1.6.5

- **The visited set of the graph search is a type, and past 2^20 vertices it is a hash table** (#119, #120).
  Up to 1.6.4 every search zeroed a bitset of `n` bits before it started: 2.9 MB of writes per search at
  23.9M vertices, and the cost that flattened large graphs (pubmed23 at 64 threads scaled 5.5× over one
  thread). `SearchGraphContext(; visited=...)` now takes the kind of set each batch slot holds:
  `BitVisited` (the bitset), `ByteVisited` (a generation byte per vertex, zeroed every 255 searches),
  `HashVisited` (an exact open-addressing table tagged with the search's generation; nothing is cleared
  between searches and the table grows with the visit, not with `n`) and `LossyHashVisited` (a fixed
  table that may forget a vertex but never reports one that was not reached; it needs a finite
  `maxvisits`). The default `AutoVisited()` is the bitset while the graph has at most `2^20` vertices and
  the table beyond, in one 128 KB buffer per slot. Same answers and evaluations; at 64 threads on a Xeon
  Silver 4216 with 8-bit codes, at the tuned point and ×0.85 / ×1.15 of its Δ, queries per second against
  the bitset: ccnews (604K) 1.00×, 0.99×, 0.99× (the table alone 0.92× and 0.82× at the two larger Δ);
  gooaq (3.0M) 1.36×, 1.30×, 1.17×; pubmed23 (23.9M) 5.7×, 3.1×, 1.5×. Construction past `2^20` vertices
  uses the table too; its time was not measured.

## 1.5

- **Multi-bit sketches.** `Projections.QuantSketch` keeps 2, 4 or 8 bits per hyperplane instead of a
  single sign bit — the same fitted model, more precision per unit of memory — and
  `Projections.SketchedSearch` packages the whole encode/index/rerank pipeline as an ordinary index.
  `index!(idx, ctx, :bitsketch; width)` bootstraps a `SearchGraph` from those wider codes.
- **Radius-bounded (ε-ball) search over `SearchGraph`.** Passing a `RadiusSorted`/`RadiusHeap`
  to `search` now navigates the graph instead of crashing, and `optimize_index!(...; radius=ε)`
  tunes the index for that workload (with `MaxMatchError`, the only error function that applies
  when a query's true ball can be empty). The answer is approximate, as any graph search is;
  `ExhaustiveSearch` remains exact.
- **Stored quantized databases can be rebuilt from their fields.** `SQu4Database`/`SQu8Database`
  accept `(E, Q)` back, so a persisted per-column database no longer has to be re-quantized from
  the `Float32` matrix it came from.
- **Quantized databases grow, over any storage.** Both families are one `ScalarQuant.QuantDatabase`
  whose codes live in any `AbstractDatabase` of `UInt8` vectors: a `BlockMatrixDatabase` or an
  `MMapMatrixDatabase` makes `push_item!`/`append_items!` quantize on the way in, so a `SearchGraph`
  builds over a quantized database one item at a time and the codes can outlive the process.
  `db.Q` is therefore a database now, not a `Matrix{UInt8}` (`db.Q.matrix` for the default
  `MatrixDatabase`); the constructors accept either. There is one `SQVec{B}` vector type and one
  set of distances (`ScalarQuant.SqL2()` and friends) for every width and both families; the
  per-width names remain as aliases. `L1` at 2 bits takes the absolute value it skipped, `NormCosine`
  exists at 4 and 2 bits, and a `GlobalQuantDatabase` rejects a dimension that does not fill its
  last byte instead of reading a plain query past its end.
- **`AsymmetricSearchGraph`.** A graph over quantized storage that inserts and searches with the raw
  objects, evaluated against the stored codes, so its edges are chosen on the exact distance; a
  `SearchGraph` over the same database is the symmetric one, codes against codes. Both are
  `AbstractSearchGraph`s, and the way of working is fixed when the instance is built. It is also the
  path for asymmetric estimators over sketches: an `AbstractEstimator` is a distance that says
  through `encode` what the storage receives and re-evaluates inside its own `evaluate` when its error
  model says it must, transparently to the graph; one serializable type, its parameters as fields.
  `ScalarQuant.Cosine` accepts a plain vector too.
- **`RaBitQ` submodule.** The RaBitQ estimator (Gao & Long, 2024) over that graph: `RaBitQCosine`/`RaBitQL2`
  store the sign bits of the rotated vector plus three scalars and evaluate a raw query against them with
  an unbiased estimate and a per-object error bound (a SIMD signed sum, 67 ns per 384-d pair);
  `RaBitQRefined` keeps a fallback beside the bits, `RaBitQExactFallback` (`Float32`/`Float16`) or
  `RaBitQVectorFallback` (scalar-quantized), and re-evaluates from it inside the estimate when the bound
  cannot rule an object out.
- **`ScalarQuant.SQEncoder`.** The scalar quantizers as the encoder of an `AsymmetricSearchGraph`, with an
  optional rotation in front: it uses the estimator interface but carries no error model, since a
  codification has no error to exploit. Its quantizer is named by the module (`SQgu4`, `SQu8`, ...)
  and its rotation by the object, `Projections.qr(dim, dim)`, the new `Projections.RandomizedHadamard`
  (random signs and the Walsh-Hadamard transform, `dim log dim`), or `nothing`; on the SISAP 2025 `ccnews`
  benchmark the rotation moved recall by less than 0.01 at every width and cost 30-50 µs per query.
- **Scores with error bars.** `bootstrapscore(recallscore, gold, res)` resamples the queries and returns the
  macro score with its standard deviation and a percentile interval, over any per-query score;
  `perqueryscores` gives the vector it draws from, and the bootstrap of the per-query differences of two
  results is the paired comparison. `matcherror` and the new `macromatcherror` are score functions in their
  own right now, exported beside `recallscore`/`macrorecall` (issue #92).
- **The Walsh-Hadamard transform is a butterfly, not an FFTW plan.** `HadamardProjection` used to build an
  FFTW plan on every per-vector `transform!`, under FFTW's global lock: 170-400 µs per vector, worse with
  threads. It is now a plain in-place butterfly on both paths, its first three passes and its `1/n` scale
  folded into `Vec{8}` operations: 0.1-4.6 µs per vector at 128-4096 dimensions (500x), 33-800 ns per
  column of a matrix over 64 threads (60x the batched FFTW call), bit for bit the same values;
  `Hadamard.jl` and FFTW leave the dependency tree (issue #89).
- **Faster quantized distances.** Per-column `SqL2`/`NormCosine` are computed from integer code
  sums rather than dequantizing coordinate by coordinate (up to 3.5x, and an order of magnitude
  more accurate), and the global `SQgu*` kernels vectorize the remainder they used to leave to
  scalar code.

## 1.4

- **`BKT`**, an exact BK-tree index for integer-valued metrics (`Levenshtein`, `DamerauLevenshtein`,
  `LCS`), built in parallel over a flat per-object workload.
- **`beginbatch`**, which lets a distance hand each batch its own scratch buffers — replacing the
  `Channel`-based pool the edit distances used, measured ~80x faster on short words.
- **`@BATCHES` accepts `:dynamic`** and uses it by default, so nested and concurrent parallel
  regions are safe.

## 1.3

- **Metric-hyperplane bit sketches**: `DistantHyperplanes`, `AnchoredDistantHyperplanes`,
  `RandomHyperplanes`, plus `PCAProjection` as a data-fitted alternative to random projections.
- **`index!(idx, ctx, :bitsketch)`**, a fast bootstrap that builds an empty `SearchGraph`'s topology
  in sketch space (`method=:gaussian`, `:qr`, `:adh`, or `:external` for precomputed sketches).
- **`MaxMatchError`**, a continuous, distance-based goal for `optimize_index!`, next to `MinRecall`.
- **`DamerauLevenshtein`**, and `String`/`SubString` accepted directly by the edit distances.

## 1.2

- **`MMapMatrixDatabase`**, a disk-backed growable database via `mmap`.
- **Two logging channels**: reporters receive progress (`INFORM`) and observers react to structural
  events (`OBSERVE`, e.g. `:add!`), so persistence can hook into an index without printing anything.
- **The `Selection` submodule**: `fft`, `dnet`, `randsel`, `multirandsel` and `neardup` under one
  roof, each returning a typed selection rather than loose arrays.

# Similarity search _ecosystem_ in Julia
Currently, there exists several packages dedicated to nearest neighbor search, for instance we have [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl), [`RegionTrees.jl`](https://github.com/rdeits/RegionTrees.jl), and [`JuliaNeighbors`](https://github.com/JuliaNeighbors) implement search structures like [kd-trees](https://en.wikipedia.org/wiki/K-d_tree), [ball trees](https://en.wikipedia.org/wiki/Ball_tree), [quadtrees](https://en.wikipedia.org/wiki/Quadtree), [octrees](https://en.wikipedia.org/wiki/Octree), [bk-trees](https://en.wikipedia.org/wiki/BK-tree), [vp-tree](https://en.wikipedia.org/wiki/Vantage-point_tree) and other multidimensional and metric structures. These structures work quite well for low dimensional data since they are designed to solve exact similarity queries.

There exist several packages performing approximate similarity search, like [`Rayuela.jl`](https://github.com/una-dinosauria/Rayuela.jl) using product quantization schemes, the wrapper for the [`FAISS`](https://faiss.ai/) library [`Faiss.jl`](https://github.com/zsz00/Faiss.jl). The FAISS library provides high-performance implementations of product quantization schemes and locality-sensitive hashing schemes, along with an industrial-strength implementation of the [`HNSW`](https://github.com/nmslib/hnswlib) index. The [`NearestNeighborDescent.jl`](https://github.com/dillondaudert/NearestNeighborDescent.jl) implements the search algorithm behind [`pynndescent`](https://pynndescent.readthedocs.io/en/latest/?badge=latest).

The `SimilaritySearch.jl` package tries to enrich the ecosystem with search structures and algorithms designed to take advantage of multithreading systems and a unique autotuning feature that simplifies its usage for practitioners. These features are succinctly and efficiently implemented due to the Julia programming language dynamism and performance.
Regarding performance characteristics, the construction times are vastly reduced compared to similar approaches without reducing search performance or result quality.

# Installing SimilaritySearch

You may install the package as follows
```julia
] add SimilaritySearch.jl
```

also, you can run the set of tests as follows
```julia
] test SimilaritySearch
```

# Using the library
Please see [examples](https://github.com/sadit/SimilaritySearchDemos). You will find a list of Jupyter and Pluto notebooks, and some scripts that exemplifies its usage.
 
# Contribute
Contributions are welcome. Please fill a pull request for documentating and implementation contributions. For issues, please fill an issue with the necessary information (see below.) If you already have a solution please also provide a pull request.

# Issues
Report issues in the package providing a minimal reproducible example. If the issue is data dependant, please don't forget to provide the necessary data to reproduce it.

## Limitations of `SearchGraph`
The main search structure, the `SearchGraph,` is a graph with several characteristics, many of them induced by the dataset being indexed. Some of its known limitations are related to these characteristics. For instance:

- Metric distances work well; on the other hand, semi-metric should work, but routing capabilities are not yet characterized.
- Even when it performs pretty well compared to alternatives, discrete metrics like Levenshtein distance and others that take few possible values may also get low performances.
- Something similar will happen when there are many near-duplicates (elements that are **pretty** close). In this case, it is necessary to remove near-duplicates and put them in _bags_ associated with some of its near objects.
- Very high dimensional datasets will produce _long-tail_ distributions of the number of edges per vertex. In extreme cases, you must prune large neighborhoods and enrich single-edge paths.

# About the structures and algorithms
The following manuscript describes and benchmarks the `SearchGraph` index (package version `0.6`):

```
@article{tellezscalable,
  title={A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs},
  author={Tellez, Eric S and Ruiz, Guillermo and Chavez, Edgar and Graff, Mario},
  journal={Pattern Analysis and Applications},
  pages={1--15},
  publisher={Springer}
}

``` 

The current algorithm (version `0.8` and `0.9`) is described and benchmarked in the following manuscript:
```

@misc{tellez2022similarity,
      title={Similarity search on neighbor's graphs with automatic Pareto optimal performance and minimum expected quality setups based on hyperparameter optimization}, 
      author={Eric S. Tellez and Guillermo Ruiz},
      year={2022},
      eprint={2201.07917},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

This package is also described in the JOSS paper:

> Eric S. Tellez and Guillermo Ruiz. _`SimilaritySearch.jl`: Autotuned nearest neighbor indexes for Julia_. Journal of Open Source Software [https://doi.org/10.21105/joss.04442](https://doi.org/10.21105/joss.04442).

## About v0.9.X series

The algorithms of this version are the same as v0.8 but break API compatibility:

- Now, it uses the `Polyester` package to handle multithreading instead of Threads.@threads
- Multithreading methods are enabled by default if the process is started with several threads; in v0.8 was the contrary
- `allknn` now preserves self-references to simplify algorithms and improve efficiency (`allknn` in v0.8 removes self-references automatically)

Others:

- Adds function docs and benchmarks
- Adds `SearchGraph` graph pruning methods
- Removes the `timedsearchbatch` function

## About v0.10.X series

It makes easy to adjust the `SearchGraph` structure to different workloads and applications. For instance,
- More control for construction parameters
- Loading and saving
- Refactors search API to be consistent across structs

Please refer to <https://github.com/sadit/SimilaritySearchDemos> and <https://github.com/sadit/SimilaritySearch.jl/blob/main/test/testsearchgraph.jl> for working examples.

## About v0.11 series

It introduces a major refactoring. In particular, it makes explicit use of context objects for most functions. It also introduces simple logging procedures.
However, we preserve compatibility in many public functions using implicit use of default context objects.

## About v0.12 series

Breaking changes:
- The context objects are now required; there are no default use of them.
- Added `ProgressMeter` for `allknn`, a small impact in the performance but pretty nice for large datasets.
- Removes dependencies of `LoopVectorization`; we only use it for `@turbo` based distance functions; these functions can be deployed in another package.

New features:
- Distances and dataset wrappers to handle non-Float32 that are casted to Float32 just before distance computations. This could improve the performance in several high throughput setups.

## About v0.15 series

Finishes a threading-model migration: every parallel loop now uses this package's own `@BATCHES` macro (a thin, native `Threads.@threads`-based construct), and the `Polyester`/`StrideArraysCore` dependencies are gone. This isn't just a simplification -- `Polyester` (and the low-level codegen machinery it and `StrideArraysCore` build on) has measurable performance regressions on Julia 1.12 and is a poor fit for static/binary deployment targets (`PackageCompiler`, WASM) that don't tolerate its runtime code-generation approach well. Native `Threads.@threads` has neither problem, which is the actual point: it's what lets this package properly support Julia 1.12+ and those deployment targets going forward.

Also in this series:
- Scalar quantization (`ScalarQuant`): reorganized into per-scheme submodules (`SQu2`/`SQu4`/`SQu8`/`SQgu4`/`SQgu8`) behind a common API, with SIMD-accelerated global 4-/8-bit quantizers.
- Random/Hadamard projections and bit sketches (`Projections`): random-rotation and Hadamard-transform dimensionality reduction, plus SimHash-style bit sketches.
- Distance functions reorganized into independent submodules under `Dist`.
- Sparse-matrix support via `SimilaritySearch.Special.Sparse`.
- Assorted bug fixes and expanded docstrings across the package.

Breaking changes:
- Removes `StrideMatrixDatabase`; use `MatrixDatabase` instead (it already accepts any `AbstractMatrix`, including a user-provided `StrideArray`).
- Removes the `StrideArraysCore` and `Polyester` dependencies (`Polyester` had already been unused internally since `@BATCHES` replaced it).

## About v1.0 series

New features:
- `apps/simsearch`: a standalone CLI application (`build`/`search`/`evaluate`/`analyze` subcommands) for working with `SimilaritySearch.jl` indexes from the shell, installable as a [Julia app](https://pkgdocs.julialang.org/dev/apps/) via `pkg> app develop apps/simsearch`.
- `Special.Spherical`: a spherical embedding (Neyshabur & Srebro) that turns Maximum Inner Product Search into ordinary nearest-neighbor search.
- A from-scratch tutorial series under `docs/src/tutorial/` (databases, distances, `SearchGraph`, whole-dataset operations, parallelism, persistence, logging), linked from the [documentation](https://sadit.github.io/SimilaritySearch.jl/dev).

Breaking changes:
- Distance/block-evaluation counters moved off the `KnnHeap`/`KnnSorted` result objects and onto the context objects (`GenericContext`/`SearchGraphContext`), tracked per parallel batch. Read them with `distance_evaluations`/`block_evaluations` (mean per batch) or `distance_stats`/`block_stats` (min/mean/max per batch); `optimize_index!`'s internal cost function was simplified accordingly.

## About v1.1 series

New features:
- `InvertedFiles`/`Intersections`: inverted-file index and posting-list intersection submodules, moved in from `TextSearch.jl` so sparse-vector, MIPS, and set search (Jaccard/Dice/Intersection/CosineSet/RogersTanimoto, or any other distance via a direct-evaluate fallback) are available directly from this package. Originally shipped as separate `BinaryInvertedFile`/`WeightedInvertedFile` types, later merged into a single `InvertedFile`.
- `Bichromatic` submodule: `bichromatic_closestpair`/`bichromatic_kclosestpairs` (the closest pair(s) between two distinct datasets) and `bichromatic_metricjoin` (a metric join between two datasets when neither a fixed radius nor a match count per element is known ahead of time). `closestpair`/`closestpairs` are now defined as the same-dataset special case of these.
- `KCenters` submodule: `fft` and `dnet` moved here, joined by two new prototype-selection algorithms, `randsel` (uniform random sampling) and `multirandsel` (randomized farthest-first, a middle ground between `randsel` and `fft`).
- `@BATCHES` scheduler control: every context now carries a `scheduler` field (`:dynamic`/`:default`/`:static`/`:greedy`/`:sequential`), so parallel loops can be forced to run single-threaded (`:sequential`) without changing `Threads.nthreads()`. The global default is `:dynamic` (`set_batch_scheduler!`/`SIMSEARCH_BATCH_SCHEDULER`): unlike `:static`, it lets unrelated indexes run their parallel loops concurrently in one process.
- A tutorial page for the `InvertedFiles`/`Bichromatic` submodules and for `ScalarQuant`/bit-sketch quantization; multi-version documentation deployment (`dev`/`stable`/`v#.#`).
- CI now runs on Julia 1.12.

Breaking changes:
- `KnnHeap`/`KnnSorted` switched to a struct-of-arrays layout (parallel `ids::Vector{UInt32}`/`dists::Vector{Float32}`, instead of a single `Vector{IdDist}`); `searchbatch`/`searchbatch!`/`allknn` now return/fill an `(ids, dists)` tuple of matrices instead of a single `Matrix{IdDist}`.
- The distance/block-evaluation counter fields were renamed to `costdists`/`costblocks`.
