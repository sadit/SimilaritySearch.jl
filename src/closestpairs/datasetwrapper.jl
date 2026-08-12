# This file is a part of SimilaritySearch.jl

"""
    bichromatic_closestpair(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; min_k::Int=8, recall::Real=1.0) -> (i, j, dist)

Convenience wrapper for [`bichromatic_closestpair`](@ref) that builds and indexes `A` and `B` itself,
mirroring [`neardup`](@ref)'s convenience-wrapper pattern: an `ExhaustiveSearch` (exact) when
`recall == 1.0` (the default), or otherwise a `SearchGraph` per dataset tuned to approach the given
`recall` via `OptimizeParameters(MinRecall(recall))`. Both `A` and `B` get the same index type and
share a single context, as [`bichromatic_closestpair`](@ref) requires.

# Arguments
- `dist`: the distance function shared by `A` and `B`
- `A`, `B`: the two datasets

# Keyword Arguments
- `min_k`: see [`bichromatic_closestpair`](@ref)
- `recall`: target recall used to decide between an exact (`recall=1.0`) or approximate index

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
    if recall < 1.0
        idxA, idxB = SearchGraph(dist, A), SearchGraph(dist, B)
        ctx = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(recall)))
        index!(idxA, ctx)
        index!(idxB, ctx)
    else
        idxA, idxB = ExhaustiveSearch(dist, A), ExhaustiveSearch(dist, B)
        ctx = GenericContext()
    end

    bichromatic_closestpair(idxA, idxB, ctx; min_k)
end
