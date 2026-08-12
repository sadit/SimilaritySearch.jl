# This file is a part of SimilaritySearch.jl

"""
    closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; min_k::Int=8) -> (i, j, dist)

Finds the closest pair among all elements indexed by `idx`. If `idx` is an approximate index then the
resulting pair may also be an approximation of the true closest pair.

Implemented as the case of [`bichromatic_closestpair`](@ref) where `idx` plays both roles -- `idxA` and
`B` (via `database(idx)`) -- which is what lets it reuse `idx`'s own internal structure (e.g. a
`SearchGraph` node's adjacency) as a search hint and excludes self-matches, without paying for a second
index.

# Arguments
- `idx`: the search structure that indexes the set of points
- `ctx`: the search context (caches, hyperparameters, etc)

# Keyword Arguments
- `min_k`: instead of looking for `k=1` some approximate methods can take advantage of a larger `k`
  (also needed for stability: must be `>= 2` here, since one slot is spent on the excluded self-match)

# Returns
A tuple `(i, j, dist)` with the identifiers `i` and `j` of the closest pair found and their distance `dist`.

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 2, 10^3))
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)

i, j, d = closestpair(G, ctx)
```
"""
function closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; min_k::Int=8)
    bichromatic_closestpair(idx, ctx, database(idx); min_k)
end

