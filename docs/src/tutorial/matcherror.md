```@meta
CurrentModule = SimilaritySearch
```

# `MaxMatchError`: A Distance-Based Alternative to `MinRecall`

[`optimize_index!`](@ref) needs some way to score a candidate `BeamSearch` configuration
while it searches for one that meets your quality target. [Tuning Search Quality](searchgraph.md#Tuning-Search-Quality:-optimize_index!)
introduced [`MinRecall`](@ref), which scores a configuration by macro-recall: the fraction
of the *exact* nearest-neighbor identifiers that the approximate search actually returned.
[`MaxMatchError`](@ref) scores it differently: instead of comparing *identifiers*, it
compares the *distances* the search actually returned against the distances of the true
nearest neighbors, rank by rank. This page explains why that distinction matters, how the
two relate in practice, and when to reach for each.

---

## The problem `MaxMatchError` addresses

`MinRecall`/[`macrorecall`](@ref) treat every returned neighbor as either a hit (its
identifier is in the exact result set) or a miss -- there is no partial credit. That is the
right notion of quality when nearby-but-wrong answers are genuinely bad, but many index
building blocks introduce *exact ties* in distance, and a "miss" that is tied in distance
with the true answer is not actually wrong:

```julia
using SimilaritySearch, Random, Statistics

# a handful of exact duplicates -> some queries have several gold neighbors tied at the same distance
Random.seed!(1)
X = randn(Float32, 8, 4000)
X[:, end-100:end] .= X[:, 1:101]
db = MatrixDatabase(X)
```

If a query's true 3rd- and 4th-nearest neighbors are two duplicate points sitting at the
*exact same distance*, an index that returns the "wrong" one of the pair in 4th place is
not making a mistake at all -- but `macrorecall` counts it as a miss regardless. This is
routine whenever a database has near-duplicate items (real corpora often do), and it's the
normal case -- not the exception -- for indexes built over a discretized proxy space, such
as a `Dist.Bits.Hamming`-compared bit sketch (used internally by
[`index!(idx, ctx, :bitsketch)`](@ref)): comparing `nbits`-bit codes only has `nbits+1`
possible distance values, so ties among candidates are the norm, not the exception.

## How `MaxMatchError` scores a result

For a query with `k' = min(k, |gold|)` true (gold) distances `d*_1 <= ... <= d*_{k'}` and the
`r` distances actually returned (both in ascending order), `MaxMatchError` computes:

```
δ_i = max(0, d_i - d*_i) / ρ      for i <= r
δ_i = η                           for i > r    (a missing position is penalized)
ρ   = (d*_{k'} - min(d*_1, d_1)) + minspread + ε
matcherror = mean(δ_i ^ p  for i in 1:k')
```

`ρ` is the gold neighborhood's own *spread*, so a `maxerror` of `0.1` means "on average,
within 10% of this query's own neighborhood spread beyond where the true answer sits" --
the same relative threshold is meaningful whether a query's neighbors happen to be tightly
clustered or spread far apart. `0` is a perfect match; there's no upper cap, so a badly-off
result keeps scoring as worse than a mildly-off one.

`minspread` exists for exactly the degenerate case described above: if a query's gold
neighbors are *all* tied (`d*_{k'} == d*_1`), the true spread is `0`, and without a real
floor `ρ` would collapse to `≈eps(Float32)` -- inflating an ordinary, non-buggy distance
difference by a factor of `10^6`-`10^7` and letting a single such query dominate a whole
batch's mean error. `minspread` (default `1f-2`) restores a sane floor instead. **Pick it
relative to your distance's own typical scale**: the default suits a `[0, 2]`-ranged
cosine-family distance, but `Dist.Bits.Hamming` over `nbits`-bit codes wants something
closer to `1f0` (one bit) -- see [`MaxMatchError`](@ref)'s docstring for the full detail.

```julia
optimize_index!(G, ctx, MaxMatchError(; maxerror=0.05f0, minspread=1f-2))
```

## Finding a `maxerror` with roughly the same bar as a `MinRecall` target

`maxerror` isn't a percentage the way `minrecall` is, so it isn't obvious up front what
value corresponds to "about as good as `MinRecall(0.9)`" on *your* data/distance. The
practical way to find out is to tune once with `MinRecall`, measure the MatchError that
configuration actually achieves, and use that as your `MaxMatchError` target:

```julia
dist = Dist.SqL2()
queries = MatrixDatabase(randn(Float32, 8, 60))
ksearch = 8

# 1. Exact gold standard (ids *and* distances -- MatchError needs the distances too)
seq = ExhaustiveSearch(dist, db)
ectx = GenericContext()
gold_ids, gold_dists = searchbatch(seq, ectx, queries, ksearch)

# 2. Tune towards a familiar MinRecall target
G = SearchGraph(dist, db)
ctx = SearchGraphContext(hyperparameters_callback=OptimizeParameters(MinRecall(0.9)))
index!(G, ctx)

# 3. Measure the MatchError *this* configuration actually achieves
knns = [knnqueue(ectx, ksearch) for _ in 1:length(queries)]
searchbatch!(G, ctx, queries, knns)
achieved = mean(SimilaritySearch.matcherror(view(gold_dists, :, i), knns[i], 1f0, 1f0)
                for i in eachindex(knns))
# achieved is now a maxerror value with roughly the same quality bar as MinRecall(0.9)
# on this dataset/distance.
```

A freshly built index tuned with `MaxMatchError(; maxerror=achieved)` will typically build
*faster* (see below) than the `MinRecall`-tuned one it was calibrated against, but its
resulting recall won't be identical -- `MinRecall` and `MaxMatchError` are two different,
only loosely related objectives, so always re-check both `macrorecall` and mean
`matcherror` after tuning rather than assuming the calibration transfers exactly.

## What actually differs in practice

| | `MinRecall` | `MaxMatchError` |
|---|---|---|
| Compares | result vs. gold **identifiers** (a set) | result vs. gold **distances**, rank by rank |
| A tied-distance "wrong" answer | scores as a full miss | scores as a near-perfect match |
| Threshold units | a recall fraction (`0`-`1`), directly interpretable | a fraction of each query's own neighborhood spread; needs calibration (see above) and a distance-appropriate `minspread` |
| Degenerate inputs | none (set membership is always well-defined) | a fully tied gold neighborhood needs `minspread` to stay well-behaved |
| Best suited for | anything, especially when a "wrong" identifier really is a wrong answer | discretized/quantized proxy spaces with frequent ties (bit sketches, scalar quantization); real data with near-duplicate items |

In repeated measurement against real, ~600k-row text embeddings (the investigation behind
[`index!(idx, ctx, :bitsketch)`](@ref)'s default `kind=MaxMatchError(; maxerror=0.01f0)`),
`MaxMatchError`-tuned construction consistently built *faster* and *far more
run-to-run-consistent* than an equivalent `MinRecall` target, while matching or exceeding
its resulting recall -- but that pattern didn't hold universally: tuning a **poorly
connected** raw topology (a `:knr` graph before its `rebuild` refinement pass) with
`MaxMatchError` performed *worse*, and with unusually high run-to-run variance, compared to
`MinRecall` on the very same graph. `MaxMatchError`'s continuous, distance-based landscape
seems to reward an already reasonably well-connected topology more reliably than
`MinRecall`'s simpler set-based one does; on a badly-connected graph, prefer `MinRecall`, or
fix the connectivity first (e.g. via [`rebuild`](@ref)).

---

Continue to [Quantization and Bit Sketches](quantization_and_bitsketches.md) for more on
building the discretized proxy spaces (bit sketches, scalar quantization) where
`MaxMatchError`'s tie-tolerance matters most.
