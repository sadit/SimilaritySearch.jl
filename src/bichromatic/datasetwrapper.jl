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
function bichromatic_closestpair(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; min_k::Int=8, recall::Real=1.0,
        verbose::Bool=false, reporters=InformativeLog(), observers=nothing)
    idxA, ctx = closestpair_buildindex(dist, A, recall; verbose, reporters, observers)
    bichromatic_closestpair(idxA, ctx, B; min_k)
end

"""
    bichromatic_kclosestpairs(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; k::Int=1, min_k::Int=max(k, 8), recall::Real=1.0) -> Vector{Tuple{Int32,Int32,Float32}}

Convenience wrapper for [`bichromatic_kclosestpairs`](@ref), analogous to
[`bichromatic_closestpair`](@ref)'s dataset wrapper above.

# Arguments
- `dist`: the distance function shared by `A` and `B`
- `A`, `B`: the two datasets (`A` gets indexed, `B` is queried directly)

# Keyword Arguments
- `k`, `min_k`: see [`bichromatic_kclosestpairs`](@ref)
- `recall`: target recall used to decide between an exact (`recall=1.0`) or approximate index

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
A = MatrixDatabase(rand(Float32, 2, 10^3))
B = MatrixDatabase(rand(Float32, 2, 10^3))

pairs = bichromatic_kclosestpairs(dist, A, B; k=10)
```
"""
function bichromatic_kclosestpairs(dist::PreMetric, A::AbstractDatabase, B::AbstractDatabase; k::Int=1, min_k::Int=max(k, 8), recall::Real=1.0,
        verbose::Bool=false, reporters=InformativeLog(), observers=nothing)
    idxA, ctx = closestpair_buildindex(dist, A, recall; verbose, reporters, observers)
    bichromatic_kclosestpairs(idxA, ctx, B; k, min_k)
end

"""
    closestpair_buildindex(dist::PreMetric, A::AbstractDatabase, recall::Real; verbose=false, reporters=InformativeLog(), observers=nothing) -> (idx, ctx)

Shared index-building step for the dataset-based convenience wrappers above: an `ExhaustiveSearch`
(exact) when `recall == 1.0`, or otherwise a `SearchGraph` tuned to approach the given `recall` via
`OptimizeParameters(MinRecall(recall))`, mirroring [`neardup`](@ref)'s convenience-wrapper pattern.
"""
function closestpair_buildindex(dist::PreMetric, A::AbstractDatabase, recall::Real;
        verbose::Bool=false, reporters=InformativeLog(), observers=nothing)
    if recall < 1.0
        idx = SearchGraph(dist, A)
        ctx = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(recall)),
                                 verbose, reporters, observers)
        index!(idx, ctx)
    else
        idx = ExhaustiveSearch(dist, A)
        ctx = GenericContext(; verbose, reporters, observers)
    end

    idx, ctx
end
