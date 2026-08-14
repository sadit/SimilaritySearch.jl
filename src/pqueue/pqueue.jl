# This file is a part of SimilaritySearch.jl

module PQueue

using ..SimilaritySearch: IdDist, IdIntDist, IdOrder, DistOrder, RevDistOrder
import ..SimilaritySearch: push_item!, reuse!, knnqueue

export AbstractMetricQueue, AbstractKnnQueue, AbstractRadiusQueue
export KnnHeap, KnnSorted, RadiusSorted, RadiusHeap, knnqueue, IdDist
export push_item!, covradius, maxlength, reuse!, viewitems, sortitems!, pop_max!, pop_min!, nearest, frontier
export DistView, IdView, IdDistView
export knn_matrices
# heap.jl's primitives and sort_last_item! are not part of the top-level SimilaritySearch export
# list (they were reachable only by explicit qualification/import before this module existed);
# exporting them here just makes them reachable as `SimilaritySearch.heapify!` etc. via
# `using .PQueue`, preserving that same qualification-only reachability without widening the
# public (unqualified `using SimilaritySearch`) interface.
export heapify!, heapsort!, isheap, heapfix_up!, heapfix_down!, sort_last_item!

"""
    AbstractMetricQueue

Abstract base type for all metric result containers. Its two direct subtypes are
[`AbstractKnnQueue`](@ref) (count-bounded: keeps the `k` closest items) and
[`AbstractRadiusQueue`](@ref) (radius-bounded: keeps every item within a fixed distance
threshold, however many that turns out to be). Both share the same underlying
`push_item!`/`nearest`/`frontier`/`viewitems` interface; only how "closest"/"kept" is bounded
differs.
"""
abstract type AbstractMetricQueue end

"""
    AbstractKnnQueue

Abstract base type for k-nearest-neighbor result containers. Concrete subtypes
([`KnnHeap`](@ref) and [`KnnSorted`](@ref)) accumulate `(id, dist)` pairs found during a
search and keep only the `k` closest ones. They share a common interface built around
[`push_item!`](@ref), [`nearest`](@ref), [`frontier`](@ref), [`viewitems`](@ref),
[`covradius`](@ref), and [`reuse!`](@ref); use [`knnqueue`](@ref) to construct one. See
[`AbstractRadiusQueue`](@ref) for the radius-bounded sibling family.
"""
abstract type AbstractKnnQueue <: AbstractMetricQueue end

"""
    AbstractRadiusQueue

Abstract base type for radius-bounded result containers ([`RadiusSorted`](@ref) and
[`RadiusHeap`](@ref)): accept an `(id, dist)` pair iff `dist <= radius`, growing without any
count limit (backed by plain growable `Vector`s, never a fixed-size or view-backed buffer).
Unlike [`AbstractKnnQueue`](@ref), [`maxlength`](@ref) always returns `typemax(Int32)` and
[`maximum`](@ref)/[`covradius`](@ref) always return the fixed `radius`, since the covering
radius is known in advance rather than discovered as the queue fills up. Construct one
directly (e.g. `RadiusSorted(radius)`); they are not wired into the `knnqueue(T, k::Int)`
capacity-based constructor since "k" has no meaning here.
"""
abstract type AbstractRadiusQueue <: AbstractMetricQueue end

@inline _lt_dist(X, i, j) = @inbounds X[2][i] < X[2][j]
@inline function _swap_ids_dists(X, i, j)
    @inbounds X[1][i], X[1][j] = X[1][j], X[1][i]
    @inbounds X[2][i], X[2][j] = X[2][j], X[2][i]
end

include("heap.jl")
include("knnheap.jl")
include("knnsorted.jl")
include("radiussorted.jl")
include("radiusheap.jl")

@inline Base.iterate(res::AbstractMetricQueue, state=1) = iterate(viewitems(res), state)

"""
    covradius(res::AbstractKnnQueue)::Float32

The covering radius of the result set, i.e., the distance to the farthest item currently
kept in `res`. While `res` has not yet reached its maximum capacity ([`maxlength`](@ref))
it returns `typemax(Float32)`, since any candidate should still be accepted.
"""
@inline covradius(res::AbstractKnnQueue)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)
@inline Base.maximum(res::AbstractKnnQueue) = frontier(res).dist
@inline Base.argmax(res::AbstractKnnQueue)  = frontier(res).id
@inline Base.minimum(res::AbstractKnnQueue) = nearest(res).dist
@inline Base.argmin(res::AbstractKnnQueue)  = nearest(res).id

# ── AbstractRadiusQueue: shared methods (RadiusSorted and RadiusHeap both store
#    ids::Vector{UInt32}, dists::Vector{Float32}, radius::Float32) ────────────────────────

@inline Base.length(res::AbstractRadiusQueue)  = length(res.ids)
@inline maxlength(::AbstractRadiusQueue)        = typemax(Int32)
@inline Base.maximum(res::AbstractRadiusQueue)  = res.radius
@inline covradius(res::AbstractRadiusQueue)     = res.radius
@inline Base.argmax(res::AbstractRadiusQueue)   = frontier(res).id
@inline Base.minimum(res::AbstractRadiusQueue)  = nearest(res).dist
@inline Base.argmin(res::AbstractRadiusQueue)   = nearest(res).id
@inline push_item!(res::AbstractRadiusQueue, i::Integer, d::Real) = push_item!(res, IdDist(convert(UInt32, i), convert(Float32, d)))
@inline push_item!(res::AbstractRadiusQueue, p::Pair) = push_item!(res, IdDist(convert(UInt32, p.first), convert(Float32, p.second)))

# ── IdView ────────────────────────────────────────────────────────────────────

"""
    IdView{ARR}

A zero-copy view over the identifier column of a collection. Indexing returns `UInt32`.

Supported wrappable types: `AbstractVector{UInt32}`, `AbstractMatrix{UInt32}`,
`AbstractVector{IdDist}`, `AbstractMatrix{IdDist}`, `KnnSorted`, `KnnHeap`.
"""
struct IdView{ARR}
    A::ARR
end

Base.length(res::IdView)   = length(res.A)
Base.size(res::IdView)     = size(res.A)
Base.eltype(::IdView)      = UInt32
Base.eltype(::Type{<:IdView}) = UInt32
Base.IteratorSize(::IdView{T}) where {T<:AbstractMatrix} = Base.HasShape{2}()
Base.IteratorSize(::IdView{T}) where {T<:AbstractVector} = Base.HasShape{1}()
Base.firstindex(res::IdView) = 1
Base.lastindex(res::IdView)  = length(res)
Base.eachindex(res::IdView)  = firstindex(res):lastindex(res)

# SoA structs: delegate to ids field
Base.getindex(res::IdView{<:KnnSorted}, i::Integer) = @inbounds res.A.ids[res.A.sp + i - 1]
Base.getindex(res::IdView{<:KnnHeap},   i::Integer) = @inbounds res.A.ids[i]
Base.getindex(res::IdView{<:AbstractRadiusQueue}, i::Integer) = @inbounds res.A.ids[i]

# Plain UInt32 arrays
Base.getindex(res::IdView{<:AbstractMatrix{UInt32}}, i...) = res.A[i...]
Base.getindex(res::IdView{<:AbstractVector{UInt32}}, i::Integer) = res.A[i]

# Legacy IdDist arrays (kept for compatibility during transition)
Base.getindex(res::IdView{<:AbstractMatrix{IdDist}}, i...) = res.A[i...].id
Base.getindex(res::IdView{<:AbstractVector{IdDist}}, i::Integer) = UInt32(res.A[i].id)
Base.getindex(res::IdView{<:AbstractVector{<:Integer}}, i::Integer) = UInt32(res.A[i])

# ── DistView ──────────────────────────────────────────────────────────────────

"""
    DistView{ARR}

A zero-copy view over the distance column of a collection. Indexing returns `Float32`.

Supported wrappable types: `AbstractVector{Float32}`, `AbstractMatrix{Float32}`,
`AbstractVector{IdDist}`, `AbstractMatrix{IdDist}`, `KnnSorted`, `KnnHeap`.
"""
struct DistView{ARR}
    A::ARR
end

Base.length(res::DistView)   = length(res.A)
Base.size(res::DistView)     = size(res.A)
Base.eltype(::DistView)      = Float32
Base.eltype(::Type{<:DistView}) = Float32
Base.IteratorSize(::DistView{T}) where {T<:AbstractMatrix} = Base.HasShape{2}()
Base.IteratorSize(::DistView{T}) where {T<:AbstractVector} = Base.HasShape{1}()
Base.firstindex(res::DistView) = 1
Base.lastindex(res::DistView)  = length(res)
Base.eachindex(res::DistView)  = firstindex(res):lastindex(res)

# SoA structs: delegate to dists field
Base.getindex(res::DistView{<:KnnSorted}, i::Integer) = @inbounds res.A.dists[res.A.sp + i - 1]
Base.getindex(res::DistView{<:KnnHeap},   i::Integer) = @inbounds res.A.dists[i]
Base.getindex(res::DistView{<:AbstractRadiusQueue}, i::Integer) = @inbounds res.A.dists[i]

# Plain Float32 arrays
Base.getindex(res::DistView{<:AbstractMatrix{Float32}}, i...) = res.A[i...]
Base.getindex(res::DistView{<:AbstractVector{Float32}}, i::Integer) = res.A[i]
Base.getindex(res::DistView{<:AbstractVector{<:AbstractFloat}}, i::Integer) = Float32(res.A[i])

# Legacy IdDist arrays
Base.getindex(res::DistView{<:AbstractMatrix{IdDist}}, i...) = res.A[i...].dist
Base.getindex(res::DistView{<:AbstractVector{IdDist}}, i::Integer) = res.A[i].dist

# Shared iterator for IdView and DistView
function Base.iterate(res::T, state::Int=1) where {T<:Union{<:IdView,<:DistView}}
    n = length(res)
    n == 0 || state > n ? nothing : (res[state], state + 1)
end

# ── IdDistView ────────────────────────────────────────────────────────────────

"""
    IdDistView{IDS, DSTS}

A lazy, zero-copy view over a range of a pair of parallel `ids`/`dists` arrays that
presents them as a sequence of [`IdDist`](@ref) pairs. Used by [`viewitems`](@ref) on
`KnnSorted` and `KnnHeap` to provide the `AbstractVector{IdDist}`-like interface without
allocating.
"""
struct IdDistView{IDS, DSTS} <: AbstractVector{IdDist}
    ids::IDS
    dists::DSTS
    sp::Int
    ep::Int
end

Base.length(v::IdDistView)   = max(0, v.ep - v.sp + 1)
Base.size(v::IdDistView)     = (length(v),)
Base.eltype(::IdDistView)    = IdDist
Base.eltype(::Type{<:IdDistView}) = IdDist
Base.firstindex(v::IdDistView) = 1
Base.lastindex(v::IdDistView)  = length(v)
Base.eachindex(v::IdDistView)  = firstindex(v):lastindex(v)

@inline function Base.getindex(v::IdDistView, i::Integer)
    j = v.sp + i - 1
    @inbounds IdDist(v.ids[j], v.dists[j])
end

function Base.iterate(v::IdDistView, state::Int=1)
    state > length(v) ? nothing : (v[state], state + 1)
end

# ── knnqueue constructors ─────────────────────────────────────────────────────

"""
    knnqueue(::Type{T}, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) where {T<:AbstractKnnQueue}

Creates a k-NN result queue of type `T` using `ids` and `dists` as its parallel backing storage.
"""
function knnqueue(::Type{T}, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) where {T<:AbstractKnnQueue}
    T(ids, dists)
end

"""
    knnqueue(::Type{T}, k::Int) where {T<:AbstractKnnQueue}

Creates a k-NN result queue of concrete type `T` (either [`KnnHeap`](@ref) or
[`KnnSorted`](@ref)) with capacity `k`, allocating fresh backing vectors of `k` zeroed
`UInt32` ids and `Float32` distances.

# Examples

```julia
res = knnqueue(KnnSorted, 3)  # capacity k = 3, freshly allocated storage
```
"""
knnqueue(::Type{T}, k::Int) where {T<:AbstractKnnQueue} =
    knnqueue(T, zeros(UInt32, k), zeros(Float32, k))

include("sparse_conversion.jl")

end
