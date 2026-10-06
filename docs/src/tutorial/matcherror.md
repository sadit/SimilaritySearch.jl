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
identifier is in the exact result set) or a miss. There is no partial credit. That is the right notion of quality when a nearby but wrong answer is genuinely bad. Many index
building blocks introduce *exact ties* in distance. A miss that is tied in distance with the true
answer is not wrong:

```julia
using SimilaritySearch, Random, Statistics

# a handful of exact duplicates -> some queries have several gold neighbors tied at the same distance
Random.seed!(1)
X = randn(Float32, 8, 4000)
X[:, end-100:end] .= X[:, 1:101]
db = MatrixDatabase(X)
```

Consider a query whose true third and fourth nearest neighbors are two duplicate points at the
*exact same distance*. An index that returns the other point of the pair in fourth place makes no
mistake. `macrorecall` counts it as a miss.

This happens whenever a database contains near-duplicate items, which real corpora often do. It
is also the normal case for an index built over a discretized proxy space. A bit sketch compared
with `Dist.Bits.Hamming` is one such space, and
[`index!(idx, ctx, :bitsketch)`](@ref) uses it internally. Comparing codes of `nbits` bits gives
only `nbits+1` possible distance values, so ties among candidates are common.

## How `MaxMatchError` scores a result

For a query with `k' = min(k, |gold|)` true (gold) distances `d*_1 <= ... <= d*_{k'}` and the
`r` distances actually returned (both in ascending order), `MaxMatchError` computes:

```
spread      = (d*_{k'} - min(d*_1, d_1)) + spreadfloor + ε
deviation_i = min(max(0, d_i - d*_i) / spread, maxdeviation)   for i <= r
deviation_i = maxdeviation                                     for i > r    (a missing position)
matcherror  = mean(deviation_i ^ exponent  for i in 1:k')
```

`spread` is the spread of the gold neighborhood itself. A `maxerror` of `0.1` therefore means
that a returned neighbor is, on average, within 10% of that spread beyond where the true answer
sits. The threshold is relative, so it keeps the same meaning whether the neighbors of a query
are tightly clustered or far apart. `0` is a perfect match. A position never costs more than `maxdeviation`, whose default is `1`. A missing position costs
the same amount. A returned neighbor that is farther than one whole spread beyond its gold
counterpart therefore counts as no neighbor at all, and the score stays within
`[0, maxdeviation ^ exponent]`. That bound is what makes the mean over queries usable. On `ccnews`, without it, ten queries whose
gold neighbors were all exact duplicates at distance `0` produced 85% of the mean over 10,500
held-out queries.

`spreadfloor` exists for the degenerate case described above. If the gold neighbors of a query
are *all* tied (`d*_{k'} == d*_1`), the true spread is `0`. Without a real floor, `spread` would
collapse to about `eps(Float32)`. An ordinary distance difference would then be multiplied by a
factor of `10^6` to `10^7`. `spreadfloor` (default `1f-2`) restores a sane floor. **Choose it relative to the typical scale of your distance.** The default suits a cosine-family
distance with a range of `[0, 2]`. `Dist.Bits.Hamming` over codes of `nbits` bits needs a value
closer to `1f0`, which is one bit. The docstring of [`MaxMatchError`](@ref) gives the full
detail.

```julia
optimize_index!(G, ctx, MaxMatchError(; maxerror=0.05f0, spreadfloor=1f-2))
```

## Finding a `maxerror` with roughly the same bar as a `MinRecall` target

`maxerror` is not a percentage, and `minrecall` is one, so it is not obvious in advance what
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

A new index tuned with `MaxMatchError(; maxerror=achieved)` usually builds *faster* than the
`MinRecall`-tuned index it was calibrated against. The section below shows this. Its recall will
not be identical, because `MinRecall` and `MaxMatchError` are two different objectives and are
only loosely related. Measure both `macrorecall` and the mean `matcherror` after tuning. Do not
assume that the calibration transfers exactly.

## What actually differs in practice

| | `MinRecall` | `MaxMatchError` |
|---|---|---|
| Compares | result vs. gold **identifiers** (a set) | result vs. gold **distances**, rank by rank |
| A tied-distance "wrong" answer | scores as a full miss | scores as a near-perfect match |
| Threshold units | a recall fraction (`0`-`1`), directly interpretable | a fraction of each query's own neighborhood spread; needs calibration (see above) and a distance-appropriate `spreadfloor` |
| Degenerate inputs | none (set membership is always well-defined) | a fully tied gold neighborhood needs `spreadfloor` to stay well-behaved |
| Best suited for | anything, especially when a "wrong" identifier really is a wrong answer | discretized/quantized proxy spaces with frequent ties (bit sketches, scalar quantization); real data with near-duplicate items |

In repeated measurement against real, ~600k-row text embeddings (the investigation behind
[`index!(idx, ctx, :bitsketch)`](@ref)'s default `kind=MaxMatchError(; maxerror=0.01f0)`),
Construction tuned with `MaxMatchError` built *faster* than construction tuned with an
equivalent `MinRecall` target, and it varied much less from one run to the next. Its recall
matched or exceeded the recall of the `MinRecall` target.

That pattern does not hold everywhere. On a **poorly connected** raw topology, a `:knr` graph
before its `rebuild` refinement pass, `MaxMatchError` performed *worse* than `MinRecall` on the
same graph, and with unusually high variance between runs.

The continuous, distance-based landscape of `MaxMatchError` appears to reward a topology that is
already reasonably well connected. The simpler set-based landscape of `MinRecall` depends on it
less. On a badly connected graph, use `MinRecall`, or repair the connectivity first with
[`rebuild`](@ref).

---

## The scores outside the optimizer, with error bars

Both goals are built on plain score functions you can call yourself. [`macrorecall`](@ref) is
the mean over the queries of [`recallscore`](@ref), and [`macromatcherror`](@ref) the mean of
[`matcherror`](@ref); `matcherror(g, r, err::MaxMatchError)` takes `exponent`, `maxdeviation` and `spreadfloor`
from the goal, so a score computed by hand is exactly the one `optimize_index!` saw.

A macro score is one number, and two indexes at 0.91 and 0.92 may or may not differ. The value depends on the sample of queries. [`bootstrapscore`](@ref) therefore resamples the
queries with replacement, over the per-query scores that [`perqueryscores`](@ref) computes once.
It returns the mean, its standard deviation, and a percentile interval:

```julia
goldI, goldD = searchbatch(ExhaustiveSearch(dist, db), GenericContext(), queries, k)
resI, _ = searchbatch(G, ctx, queries, k)
bootstrapscore(recallscore, goldI, resI)
# BootstrapScore(0.9155 ± 0.0089, 95% [0.898, 0.9325], 200 queries, 1000 resamples)

knns = [search(G, ctx, queries[i], knnqueue(KnnSorted, k)) for i in eachindex(queries)]
bootstrapscore((g, r) -> matcherror(g, r, MaxMatchError()), goldD, knns)
```

Two indexes measured on the **same** queries are compared in pairs. The bootstrap resamples the
per-query differences, so every draw takes the same queries from both indexes. An interval that
excludes zero is the evidence that the two indexes differ at that level:

```julia
a = perqueryscores(recallscore, goldI, resA)
b = perqueryscores(recallscore, goldI, resB)
bootstrapscore(a .- b; nboot=10_000)
```

---

Continue to [Quantization and Bit Sketches](quantization_and_bitsketches.md) for more on
building the discretized proxy spaces (bit sketches, scalar quantization) where
`MaxMatchError`'s tie-tolerance matters most.
