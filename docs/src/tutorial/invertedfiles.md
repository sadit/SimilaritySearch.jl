```@meta
CurrentModule = SimilaritySearch
```

# Inverted Files and Posting List Intersections

Inverted indexes store posting lists mapping component dimensions (or set elements/tokens) to document identifiers and weights. They enable fast exact and approximate search over high-dimensional sparse data, set collections, and Maximum Inner Product Search (MIPS) workloads.

`SimilaritySearch.jl` exposes inverted indexes in the `InvertedFiles` submodule (`SimilaritySearch.InvertedFiles`) and lower-level posting list intersection algorithms in `Intersections` (`SimilaritySearch.Intersections`).

## `InvertedFiles` Overview

`InvertedFiles` provides two primary index types:

1. **[`InvertedFiles.WeightedInvertedFile`](@ref)** — For sparse vector search, Maximum Inner Product Search (MIPS), and normalized cosine similarity search.
2. **[`InvertedFiles.BinaryInvertedFile`](@ref)** — For set collections evaluated with set metrics like `Dist.Sets.Jaccard()`, `Dist.Sets.Dice()`, or `Dist.Sets.Intersection()`.

Both inherit from `AbstractSearchIndex` and implement the standard `SimilaritySearch` interface (`append_items!`, `push_item!`, `search`, `searchbatch`).

---

## Worked Example: `WeightedInvertedFile`

`WeightedInvertedFile` indexes items represented as sparse vectors (or key-weight mappings like `Dict` or `SparseVector`).

```julia
using SimilaritySearch
using SimilaritySearch.InvertedFiles

# Dimension (vocabulary size) = 1000
vocsize = 1000
invfile = WeightedInvertedFile(vocsize)
ctx = InvertedFileContext()

# Create sample sparse vectors
items = [
    Dict(1 => 0.5f0, 10 => 0.8f0, 100 => 0.1f0),
    Dict(2 => 0.9f0, 10 => 0.4f0, 500 => 0.3f0),
    Dict(1 => 0.6f0, 20 => 0.7f0, 100 => 0.4f0),
]
db = VectorDatabase(items)

# Populate inverted file index
append_items!(invfile, ctx, db)

# Querying the index
q = Dict(1 => 0.7f0, 100 => 0.3f0)
res = knnqueue(ctx, 2)
search(invfile, ctx, q, res)

for p in viewitems(res)
    println("id=", p.id, " dist=", p.dist)
end
```

---

## Worked Example: `BinaryInvertedFile`

`BinaryInvertedFile` indexes set items represented as sorted arrays of integer identifiers, matching set distance functions such as `Dist.Sets.Jaccard()`.

```julia
using SimilaritySearch
using SimilaritySearch.InvertedFiles

vocsize = 100
dist = Dist.Sets.Jaccard()
binvfile = BinaryInvertedFile(vocsize, dist)
ctx = InvertedFileContext()

sets = [
    UInt32[1, 5, 12, 30],
    UInt32[2, 5, 12, 45],
    UInt32[1, 12, 30, 80]
]
db = VectorDatabase(sets)

append_items!(binvfile, ctx, db)

query_set = UInt32[1, 12, 30]
res = knnqueue(ctx, 2)
search(binvfile, ctx, query_set, res)
```

---

## Posting List Intersection Algorithms (`Intersections`)

The `Intersections` submodule provides posting list intersection routines used internally by `InvertedFiles`:

- `Intersections.svs` — Smallest-Vector-First intersection.
- `Intersections.bk!` / `Intersections.bkt!` — Baeza-Yates / WAND-style candidate threshold algorithms.
- `Intersections.umerge!` / `Intersections.imerge!` — Union and intersection merges across multiple posting lists.
- `Intersections.xmerge!` — General threshold-based multi-way list merge.

```julia
using SimilaritySearch.Intersections

list1 = [1, 3, 5, 7, 9]
list2 = [1, 2, 3, 4, 5]

# SVS intersection
common = svs([list1, list2])
```
