```@meta
CurrentModule = SimilaritySearch
```

# Inverted Files and Posting List Intersections

Inverted indexes store posting lists mapping component dimensions (or set elements/tokens) to document identifiers. They enable fast exact search over high-dimensional sparse data, set collections, and Maximum Inner Product Search (MIPS) workloads.

`SimilaritySearch.jl` exposes inverted indexes in the `InvertedFiles` submodule (`SimilaritySearch.InvertedFiles`) and lower-level posting list intersection algorithms in `Intersections` (`SimilaritySearch.Intersections`).

## `InvertedFiles` Overview

There is a single index type and a single constructor: [`InvertedFiles.InvertedFile(vocsize, dist)`](@ref)
— plain token/set-membership posting lists (`AdjType`'s element type is always `UInt32`). The
distance `dist` alone decides how candidates get scored:

1. For a handful of distances (the five set metrics: `Dist.Sets.Jaccard()`, `Dist.Sets.Dice()`,
   `Dist.Sets.Intersection()`, `Dist.Sets.CosineSet()`, `Dist.Sets.RogersTanimoto(σ)`), the score
   computed while merging posting lists is already the exact distance — no extra work needed.
2. For any other distance — including `Dist.NormCosine()`, used below for sparse-vector search,
   MIPS, and cosine similarity — `search` evaluates `dist` directly against the objects stored in
   `database(idx)` for every merge candidate, so results stay exact. This is what lets `InvertedFile`
   support distances with no posting-list-friendly closed form — e.g. a sequence-edit distance over
   shared-token candidates — with no new indexing machinery: as long as your object type has an
   [`identiterator`](@ref InvertedFiles.identiterator) method (see the last section), you can plug
   in `evaluate(mydist, a, b)` and get a working, exact-per-candidate search out of the box. The
   `t`-threshold parameter of `search` (see below) doubles as this path's cost knob: raise it above
   the default `1` to reduce how many real `evaluate` calls happen per query.

`InvertedFile` always supports the standard `SimilaritySearch` interface (`append_items!`,
`push_item!`, `search`, `searchbatch`), and always keeps a copy of every indexed object in
`database(idx)`.

---

## Worked example: which recipe uses these ingredients?

We'll index a handful of recipes twice — once as plain ingredient *sets* (AND/OR/Jaccard/RogersTanimoto), once as ingredient *quantity* vectors (`Dist.NormCosine()`) — and answer the same kind of question five different ways.

```julia
using SimilaritySearch, SimilaritySearch.InvertedFiles

ingredients = ["flour", "sugar", "egg", "butter", "milk", "salt",
               "yeast", "tomato", "cheese", "basil", "chicken", "rice"]
id(name) = findfirst(==(name), ingredients)
vocsize = length(ingredients)   # 12

recipe_sets = Dict(
    "Pancakes"        => UInt32[id("flour"), id("sugar"), id("egg"), id("milk")],
    "Bread"           => UInt32[id("flour"), id("salt"), id("yeast")],
    "MargheritaPizza" => UInt32[id("flour"), id("tomato"), id("cheese"), id("basil")],
    "Omelette"        => UInt32[id("egg"), id("butter"), id("milk"), id("cheese")],
    "ChickenRice"     => UInt32[id("chicken"), id("rice"), id("salt")],
    "Cheesecake"      => UInt32[id("sugar"), id("egg"), id("butter"), id("cheese")],
)
names = collect(keys(recipe_sets))
sets = VectorDatabase([sort!(recipe_sets[n]) for n in names])

ctx = InvertedFileContext()
q = UInt32[id("flour"), id("cheese")]   # "what uses flour AND/OR cheese?"
```

### AND query

`t` is the posting-list merge threshold: `t = length(Q)` requires *every* query token to match (an intersection/AND), where `Q` is the number of distinct query tokens actually found in the vocabulary.

```julia
IJ = InvertedFile(vocsize, Dist.Sets.Jaccard())
append_items!(IJ, ctx, sets)

res = knnqueue(ctx, 6)
search(IJ, ctx, q, res; t=length(q))
for it in viewitems(res)
    println(names[it.id], " => ", it.dist)
end
# MargheritaPizza => 0.5
```

Only `MargheritaPizza` contains *both* flour and cheese, so the AND query returns exactly one match.

### OR query

`t=1` is a union: any query token matching is enough.

```julia
res = knnqueue(ctx, 6)
search(IJ, ctx, q, res; t=1)
for it in viewitems(res)
    println(names[it.id], " => ", it.dist)
end
# MargheritaPizza => 0.5
# Bread           => 0.75
# Pancakes        => 0.8
# Cheesecake      => 0.8
# Omelette        => 0.8
```

Every recipe sharing at least one of {flour, cheese} shows up, ranked by `IJ`'s distance (`Dist.Sets.Jaccard()` here) rather than filtered out — this is what the rest of the sections vary: same OR-style candidate generation, different distance for ranking.

### Jaccard ranking

The numbers above *are* the Jaccard ranking (`IJ`'s distance is `Dist.Sets.Jaccard()`) — this score is exact, computed purely from `|q ∩ item|` and the two set sizes, no `rerank!` involved: `Pancakes`/`Cheesecake`/`Omelette` tie exactly at `0.8` because Jaccard only sees that each shares exactly one of the two query ingredients out of four total, same as the prime-factor ties in [A gallery of distances](distances.md).

### RogersTanimoto ranking

`RogersTanimoto(σ)` additionally credits recipes that *agree on ingredients neither uses* — `σ` is the size of the full ingredient universe (`vocsize` here):

```julia
IR = InvertedFile(vocsize, Dist.Sets.RogersTanimoto(vocsize))
append_items!(IR, ctx, sets)

res = knnqueue(ctx, 6)
search(IR, ctx, q, res; t=1)
for it in viewitems(res)
    println(names[it.id], " => ", it.dist)
end
# MargheritaPizza => 0.2857143
# Bread           => 0.4
# Pancakes        => 0.5
# Cheesecake      => 0.5
# Omelette        => 0.5
```

Same ranking order as Jaccard here, but note the *gap* between the top two changes (`0.5`/`0.75` under Jaccard vs. `0.286`/`0.4` under RogersTanimoto) — with only 12 ingredients in the universe, shared absences carry real weight. Like the four set metrics above it, this is also an exact fast path: no extra evaluation against `database(idx)` needed.

### NormCosine ranking

For a *quantity*-sensitive answer (not just "does it contain X"), index each recipe as an L2-normalized ingredient-quantity vector and build an `InvertedFile` with `Dist.NormCosine()`:

```julia
using SparseArrays, LinearAlgebra

l2normalize(idx::Vector{<:Integer}, val::Vector{<:Real}, n) = normalize!(sparsevec(idx, Float32.(val), n))

recipe_weights = Dict(
    "Pancakes"        => l2normalize([id("flour"),id("sugar"),id("egg"),id("milk")], [3.0,1.0,2.0,2.0], vocsize),
    "Bread"           => l2normalize([id("flour"),id("salt"),id("yeast")], [4.0,1.0,1.0], vocsize),
    "MargheritaPizza" => l2normalize([id("flour"),id("tomato"),id("cheese"),id("basil")], [3.0,2.0,2.0,1.0], vocsize),
    "Omelette"        => l2normalize([id("egg"),id("butter"),id("milk"),id("cheese")], [3.0,1.0,1.0,2.0], vocsize),
    "ChickenRice"     => l2normalize([id("chicken"),id("rice"),id("salt")], [2.0,2.0,1.0], vocsize),
    "Cheesecake"      => l2normalize([id("sugar"),id("egg"),id("butter"),id("cheese")], [2.0,2.0,1.0,3.0], vocsize),
)
weights = VectorDatabase([recipe_weights[n] for n in names])

W = InvertedFile(vocsize, Dist.NormCosine())
append_items!(W, ctx, weights)

qw = l2normalize([id("flour"),id("egg"),id("cheese")], [1.0,1.0,1.0], vocsize)
res = knnqueue(ctx, 6)
search(W, ctx, qw, res)
for it in viewitems(res)
    println(names[it.id], " => ", it.dist)
end
# Omelette        => 0.2546
# Pancakes        => 0.3196
# Cheesecake      => 0.3196
# MargheritaPizza => 0.3196
# Bread           => 0.4557
```

`MargheritaPizza` was the clear AND/Jaccard/RogersTanimoto winner, but under NormCosine `Omelette` (which weights egg and cheese heavily, matching the query's emphasis on egg/cheese/flour in *proportion*, not just presence) comes out ahead — same question, a genuinely different notion of "closest" depending on whether presence or proportion is what matters, echoing the same point [A gallery of distances](distances.md) makes about `Jaccard` vs. `Levenshtein`.

### Why three index instances?

`IJ`, `IR`, and `W` are all `InvertedFile` — built with the same constructor, differing only in the
`dist` passed at construction time. An index is tied to one distance for its lifetime, so answering
the same question under a different notion of "closest" means building a separate instance, not
reconfiguring an existing one.

---

## Extending `InvertedFile` to new object types and distances

`database(idx)` always holds the original objects exactly as given — no canonical encoding is imposed, so a `db` can freely mix `Set`s, sorted `Vector`s, `Dict`s, sparse vectors, or any other type your distance's `evaluate` accepts. `InvertedFile` itself never needs a weight to build or query posting lists — the handful of distances with an exact fast path score from intersection/set sizes alone, and every other distance is evaluated directly against the full objects kept in `db` — so the piece that turns a native object into an id stream for building/querying posting lists is [`InvertedFiles.identiterator`](@ref); `SparseArrays.SparseVector` and this package's `Special.Sparse.SparseVecView` are both handled efficiently out of the box (iterating only the non-zero indices), which is what `Dist.NormCosine()`/`Cosine()`/`Angle()`/`NormAngle()` expect — those distances have no `evaluate` method for `Dict`. A plain dense `Vector` is *not* accepted by `identiterator` — convert it first (e.g. `SparseArrays.sparse(v)`) so the reduction to non-zero components is explicit in your own code rather than happening silently.

```julia
identiterator(dist::PreMetric, obj)
```

which defaults to the distance-agnostic `identiterator(obj)` dispatch already used above (covering `Set`, sorted integer vectors, `Dict`s, dense/sparse vectors). Overload it for a specific `(DistType, ObjType)` pair when the same native object type needs to generate different candidate ids depending on which distance the index is built for — for instance, a sequence distance (e.g. `Dist.Seqs.Levenshtein`) could plug in this way, generating candidates from shared tokens/shingles and letting the generic direct-evaluate fallback compute the true edit distance over the stored raw sequences. This package does not ship string/q-gram tokenization support out of the box — that's tracked as a follow-up in [`TextSearch.jl`](https://github.com/sadit/TextSearch.jl), which builds on `InvertedFile` for exactly that use case.

---

## Posting List Intersection Algorithms (`Intersections`)

The `Intersections` submodule provides posting list intersection routines used internally by `InvertedFiles`:

- `Intersections.svs` — Smallest-Vector-First intersection.
- `Intersections.bk!` / `Intersections.bkt!` — Baeza-Yates / WAND-style candidate threshold algorithms.
- `Intersections.umerge!` / `Intersections.imerge!` — Union and intersection merges across multiple posting lists.
- `Intersections.xmerge!` — General threshold-based multi-way list merge (this is what the `t` keyword above ultimately dispatches to).

```julia
using SimilaritySearch.Intersections

list1 = [1, 3, 5, 7, 9]
list2 = [1, 2, 3, 4, 5]

# SVS intersection
common = svs([list1, list2])
```

Next: [Quantization and Bit Sketches](quantization_and_bitsketches.md).
