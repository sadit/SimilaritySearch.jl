```@meta
CurrentModule = SimilaritySearch
```

# Sketched Search: the Whole Pipeline as an Index

Searching with sketches is always the same four steps: encode the dataset, index the codes,
retrieve more candidates than asked for under the cheap sketch distance, and re-score those
candidates with the real one. Written by hand it is a handful of lines --
[`Projections.SketchedSearch`](@ref) exists because every one of those lines is a place to be
silently wrong:

- encoding the query with a **freshly fitted** model instead of the dataset's, which produces
  codes that are not comparable with the stored ones;
- indexing the sketches under the **original** distance instead of `distance(qs)`;
- forgetting the re-scoring pass and returning sketch distances as if they were real ones.

None of those raise an error. They return worse results, which is the hardest kind of bug to
notice in an approximate search.

---

## Using it

`SketchedSearch` is an ordinary [`AbstractSearchIndex`](@ref): [`search`](@ref) and
[`searchbatch`](@ref) work on it unchanged, and the identifiers it returns are indices into
the **original** database, with true distances -- so swapping it in is the whole experiment.

```julia
# SimilaritySearch v1.5
using SimilaritySearch
const P = SimilaritySearch.Projections

X = randn(Float32, 128, 5_000)
db = MatrixDatabase(X)
dist = Dist.SqL2()

model = P.gaussian(128, 256)                       # 256 hyperplanes
S = P.SketchedSearch(model, 4, dist, db; factor=8) # 4 bits each, 8x candidates before re-scoring

ctx = GenericContext()
q = randn(Float32, 128)
res = search(S, ctx, q, knnqueue(KnnSorted, 10))

for p in IdDistView(res)
    println("id=", p.id, "  distance=", p.dist)    # ids into `db`, distances under `dist`
end
```

Compare against the exact answer to see what the sketch stage costs:

```julia
# SimilaritySearch v1.5
using SimilaritySearch
const P = SimilaritySearch.Projections

X = randn(Float32, 128, 5_000)
db = MatrixDatabase(X)
dist = Dist.SqL2()
Q = MatrixDatabase(randn(Float32, 128, 30))
ctx = GenericContext()

gold, _ = searchbatch(ExhaustiveSearch(dist, db), ctx, Q, 10)

for factor in (1, 4, 16)
    S = P.SketchedSearch(P.gaussian(128, 256), 4, dist, db; factor)
    ids, _ = searchbatch(S, ctx, Q, 10)
    println("factor=", factor, "  recall=", macrorecall(gold, ids))
end
```

---

## The knobs, and what each one trades

| knob | what it does | what it costs |
| :--- | :--- | :--- |
| `nbits` (2nd positional) | bits per hyperplane: `1` is a plain bit sketch with Hamming, `2`/`4`/`8` are [`Projections.QuantSketch`](@ref) codes | memory, linearly |
| `factor` | how many times `k` candidates the sketch stage retrieves before re-scoring | time in the re-scoring pass |
| `index` | how the sketches themselves are indexed; defaults to [`Projections.exhaustivesketchindex`](@ref) | build time vs sketch-stage speed |

`factor` is the recall knob: the sketch stage is fast but lossy, so it is asked for more
candidates than needed and the exact distance settles the final order among them. `factor=1`
disables the widening but still re-scores, so the distances stay exact even then.

---

## What it is not

`SketchedSearch` is built once over a fixed database and is **not incremental**. Its encoder's
quantization range -- and, for the hyperplane models, its anchors -- are fitted on exactly the
data given to the constructor, so growing the collection means rebuilding it rather than
pushing into it. For a growing collection, bootstrap a [`SearchGraph`](@ref) from sketches
instead (see [Multi-Bit Sketches](multibit_sketches.md)) and keep inserting into the graph.

This closes the tutorial series. Sketches are the last of the compression strategies; for
queries bounded by a distance rather than by a count, see
[Radius Queries](radius_search.md), and for the full API, the [reference](../api.md).
