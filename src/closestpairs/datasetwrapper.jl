# This file is a part of SimilaritySearch.jl

"""
    bichromatic_closestpair(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; min_k::Int=8, recall::Real=1.0) -> (i, j, dist)

Convenience wrapper for [`bichromatic_closestpair`](@ref) that builds and indexes `A` and `B` itself,
mirroring [`neardup`](@ref)'s convenience-wrapper pattern: an `ExhaustiveSearch` (exact) when
`recall == 1.0` (the default), or otherwise a `SearchGraph` per dataset tuned to approach the given
`recall` via `OptimizeParameters(MinRecall(recall))`.

# Arguments
- `dist`: the distance function shared by `A` and `B`
- `A`, `B`: the two datasets

# Keyword Arguments
- `min_k`: see [`bichromatic_closestpair`](@ref)
- `recall`: target recall used to decide between an exact (`recall=1.0`) or approximate index per dataset

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
A = MatrixDatabase(rand(Float32, 2, 10^3))
B = MatrixDatabase(rand(Float32, 2, 10^3))

i, j, d = bichromatic_closestpair(dist, A, B)
```
"""
function bichromatic_closestpair(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; min_k::Int=8, recall::Real=1.0)
    idxA, ctxA = closestpair_buildindex(dist, A, recall)
    idxB, ctxB = closestpair_buildindex(dist, B, recall)
    bichromatic_closestpair(idxA, ctxA, idxB, ctxB; min_k)
end

function closestpair_buildindex(dist::PreMetric, X::AbstractDatabase, recall::Real)
    if recall < 1.0
        idx = SearchGraph(dist, X)
        ctx = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(recall)))
        index!(idx, ctx)
    else
        idx = ExhaustiveSearch(dist, X)
        ctx = GenericContext()
    end

    idx, ctx
end
