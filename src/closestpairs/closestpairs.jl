# This file is a part of SimilaritySearch.jl

"""
    closestpairs(idx::AbstractSearchIndex, ctx::AbstractContext; k::Int=1, min_k::Int=max(k, 8)) -> Vector{Tuple{Int32,Int32,Float32}}

Finds the `k` closest pairs among all elements indexed by `idx`. If `idx` is an approximate index then
the resulting pairs may also be an approximation of the true `k` closest pairs.

Implemented as the case of [`bichromatic_kclosestpairs`](@ref) where `idx` plays both roles -- `idxA` and
`B` (via `database(idx)`) -- exactly as [`closestpair`](@ref) does for a single pair (`k == 1`).

# Arguments
- `idx`: the search structure that indexes the set of points
- `ctx`: the search context (caches, hyperparameters, etc)

# Keyword Arguments
- `k`: how many globally closest pairs to return
- `min_k`: see [`bichromatic_kclosestpairs`](@ref); must be `>= k` for exactness (the default
  `max(k, 8)` guarantees this)

# Returns
Up to `k` tuples `(i, j, dist)` with the identifiers of each closest pair and their distance, sorted
ascending by distance. Fewer than `k` tuples are returned if `idx` has fewer than `k` eligible pairs.

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 2, 10^3))
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)

pairs = closestpairs(G, ctx; k=10)
```
"""
function closestpairs(idx::AbstractSearchIndex, ctx::AbstractContext; k::Int=1, min_k::Int=max(k, 8))
    bichromatic_kclosestpairs(idx, ctx, database(idx); k, min_k)
end
