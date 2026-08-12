# This file is a part of SimilaritySearch.jl

"""
    closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; min_k::Int=8) -> (i, j, dist)

Finds the closest pair among all elements indexed by `idx`. If `idx` is an approximate index then the
resulting pair may also be an approximation of the true closest pair.

Implemented as the monochromatic case of [`bichromatic_closestpair`](@ref) -- i.e., `idx` is passed as
both sides of the pair, which is what lets it reuse `idx`'s own internal structure (e.g. a `SearchGraph`
node's adjacency) as a search hint and excludes self-matches, without paying for a second index.

# Arguments
- `idx`: the search structure that indexes the set of points
- `ctx`: the search context (caches, hyperparameters, etc)

# Keyword Arguments
- `min_k`: instead of looking for `k=1` some approximate methods can take advantage of a larger `k`

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
    bichromatic_closestpair(idx, ctx, idx, ctx; min_k)
end

function search_hint(idx::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    res = search(idx, ctx, database(idx, i), res)
    if argmin(res) == i
        pop_min!(res)
    end

    nearest(res)
end

function search_hint(G::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    vstate = getvstate(length(G), ctx)
    visit!(vstate, convert(UInt64, i))
    res = search(G.algo[], G, ctx, database(G, i), res, rand(neighbors(G.adj, i)), vstate)
    if argmin(res) == i
        pop_min!(res)
    end

    nearest(res)
end
