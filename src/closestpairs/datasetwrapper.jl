# This file is a part of SimilaritySearch.jl

"""
    bichromatic_closestpair(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; min_k::Int=8, recall::Real=1.0) -> (i, j, dist)

Convenience wrapper for [`bichromatic_closestpair`](@ref) that builds and indexes `A` itself (`B` is
queried directly, unindexed, as [`bichromatic_closestpair`](@ref) does), mirroring [`neardup`](@ref)'s
convenience-wrapper pattern: an `ExhaustiveSearch` (exact) when `recall == 1.0` (the default), or
otherwise a `SearchGraph` tuned to approach the given `recall` via `OptimizeParameters(MinRecall(recall))`.

# Arguments
- `dist`: the distance function shared by `A` and `B`
- `A`, `B`: the two datasets (`A` gets indexed, `B` is queried directly)

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
        idxA = SearchGraph(dist, A)
        ctx = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(recall)))
        index!(idxA, ctx)
    else
        idxA = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
    end

    bichromatic_closestpair(idxA, ctx, B; min_k)
end
