# This file is a part of SimilaritySearch.jl

#module KnnResult

# export AbstractKnnQueueesult
export AbstractKnn, KnnHeap, KnnSorted, knnqueue, IdDist
export push_item!, covradius, maxlength, reuse!, viewitems, sortitems!, pop_max!, nearest, frontier
export DistView, IdView
export distance_evaluations, block_evaluations

"""
    AbstractKnn

Abstract base type for k-nearest-neighbor result containers. Concrete subtypes
([`KnnHeap`](@ref) and [`KnnSorted`](@ref)) accumulate `(id, dist)` pairs found during a
search and keep only the `k` closest ones. They share a common interface built around
[`push_item!`](@ref), [`nearest`](@ref), [`frontier`](@ref), [`viewitems`](@ref),
[`covradius`](@ref), and [`reuse!`](@ref); use [`knnqueue`](@ref) to construct one.
"""
abstract type AbstractKnn end

#=struct IdDist
    id::UInt32
    dist::Float32
end=#

#using Base.Order
#import Base.Order: lt
#
#struct WeightOrderingType <: Ordering end
#struct RevWeightOrderingType <: Ordering end
#const DistOrder = WeightOrderingType()
#const RevDistOrder = RevWeightOrderingType()

##@inline lt(::WeightOrderingType, a::IdDist, b::IdDist) = a.dist < b.dist
##@inline lt(::RevWeightOrderingType, a::IdDist, b::IdDist) = b.dist < a.dist
##@inline lt(::WeightOrderingType, a::Number, b::Number) = a < b
##@inline lt(::RevWeightOrderingType, a::Number, b::Number) = b < a

include("heap.jl")
include("knnheap.jl")
include("knnsorted.jl")

"""
    covradius(res::AbstractKnn)::Float32

The covering radius of the result set, i.e., the distance to the farthest item currently
kept in `res`. While `res` has not yet reached its maximum capacity ([`maxlength`](@ref))
it returns `typemax(Float32)`, since any candidate should still be accepted.
"""
@inline covradius(res::AbstractKnn)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)
@inline Base.maximum(res::AbstractKnn) = frontier(res).dist
@inline Base.argmax(res::AbstractKnn) = frontier(res).id
@inline Base.minimum(res::AbstractKnn) = nearest(res).dist
@inline Base.argmin(res::AbstractKnn) = nearest(res).id

Base.convert(::Type{T}, v::IdDist) where {T<:Integer} = convert(T, v.id)
Base.convert(::Type{T}, v::IdDist) where {T<:AbstractFloat} = convert(T, v.dist)
Base.convert(::Type{T}, v::AbstractVector{IdDist}) where {T<:Vector{<:Integer}} = T(IdView(v))
Base.convert(::Type{T}, v::AbstractVector{IdDist}) where {T<:Vector{<:AbstractFloat}} = T(DistView(v))
function Base.convert(::Type{T}, v::AbstractMatrix{IdDist}) where {T<:Matrix{<:Integer}}
    X = T(undef, size(v))
    V = IdView(v)
    for i in eachindex(X)
        X[i] = V[i]
    end
    X
end

function Base.convert(::Type{T}, v::AbstractMatrix{IdDist}) where {T<:Matrix{<:AbstractFloat}}
    X = T(undef, size(v))
    V = DistView(v)
    for i in eachindex(X)
        X[i] = V[i]
    end
    X
end

"""
    IdView{ARR}

A zero-copy view over the identifier column of a collection of [`IdDist`](@ref) items
(e.g., a [`KnnHeap`](@ref)/[`KnnSorted`](@ref) result set, or an `AbstractVector`/
`AbstractMatrix{IdDist}`). Indexing an `IdView` returns the `id` field of the
corresponding item instead of the full `IdDist`.

# Fields
- `A`: the underlying collection being viewed.
"""
struct IdView{ARR}
    A::ARR
end

Base.length(res::IdView) = length(res.A)
Base.size(res::IdView) = size(res.A)
Base.eltype(::IdView) = UInt32
Base.eltype(::Type{<:IdView}) = UInt32
Base.IteratorSize(::IdView{T}) where {T<:AbstractMatrix} = Base.HasShape{2}()
Base.IteratorSize(::IdView{T}) where {T<:AbstractVector} = Base.HasShape{1}()
Base.firstindex(res::IdView) = 1
Base.lastindex(res::IdView) = length(res)
Base.eachindex(res::IdView) = firstindex(res):lastindex(res)
Base.getindex(res::IdView{<:AbstractMatrix{IdDist}}, i...) = res.A[i...].id
Base.getindex(res::IdView{<:AbstractVector{IdDist}}, i::Integer) = UInt32(res.A[i].id)
Base.getindex(res::IdView{<:AbstractVector{<:Integer}}, i::Integer) = UInt32(res.A[i])
Base.getindex(res::IdView{<:KnnHeap}, i::Integer) = res.A.items[i].id
Base.getindex(res::IdView{<:KnnSorted}, i::Integer) = res.A.items[res.A.sp+i-1].id

"""
    DistView{ARR}

A zero-copy view over the distance column of a collection of [`IdDist`](@ref) items
(e.g., a [`KnnHeap`](@ref)/[`KnnSorted`](@ref) result set, or an `AbstractVector`/
`AbstractMatrix{IdDist}`). Indexing a `DistView` returns the `dist` field of the
corresponding item instead of the full `IdDist`.

# Fields
- `A`: the underlying collection being viewed.
"""
struct DistView{ARR}
    A::ARR
end

Base.length(res::DistView) = length(res.A)
Base.eltype(::DistView) = Float32
Base.IteratorSize(::DistView{T}) where {T<:AbstractMatrix} = Base.HasShape{2}()
Base.IteratorSize(::DistView{T}) where {T<:AbstractVector} = Base.HasShape{1}()
Base.eltype(::Type{<:DistView}) = Float32
Base.size(res::DistView) = size(res.A)
Base.firstindex(res::DistView) = 1
Base.lastindex(res::DistView) = length(res)
Base.eachindex(res::DistView) = firstindex(res):lastindex(res)
Base.getindex(res::DistView{<:AbstractMatrix{IdDist}}, i...) = res.A[i...].dist
Base.getindex(res::DistView{<:AbstractVector{IdDist}}, i::Integer) = res.A[i].dist
Base.getindex(res::DistView{<:AbstractVector{<:AbstractFloat}}, i::Integer) = Float32(res.A[i])
Base.getindex(res::DistView{<:KnnHeap}, i::Integer) = res.A.items[i].dist
Base.getindex(res::DistView{<:KnnSorted}, i::Integer) = res.A.items[res.A.sp+i-1].dist

function Base.iterate(res::T, state::Int=1) where {T<:Union{<:IdView,<:DistView}}
    n = length(res)
    if n == 0 || state > n
        nothing
    else
        res[state], state + 1
    end
end


"""
    knnqueue(::Type{KnnHeap}, vec::AbstractVector)

Creates a [`KnnHeap`](@ref) k-NN result queue using `vec` as its initial backing storage
(its length sets the capacity `k`). The queue starts with zero items and grows with
[`push_item!`](@ref) calls until capacity `k` is reached; after that, only the closest
items (by distance) are preserved.

# Examples

```julia
res = knnqueue(KnnHeap, 3)   # capacity k = 3
push_item!(res, 1, 0.5f0)
push_item!(res, 2 => 0.1f0)
nearest(res)
```
"""
knnqueue(::Type{KnnHeap}, vec::AbstractVector) = KnnHeap(vec, zero(IdDist), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))

"""
    knnqueue(::Type{KnnSorted}, vec::AbstractVector)

Creates a [`KnnSorted`](@ref) k-NN result queue using `vec` as its initial backing storage
(its length sets the capacity `k`). Behaves like `knnqueue(KnnHeap, vec)`, but keeps its
active items always sorted by distance.
"""
knnqueue(::Type{KnnSorted}, vec::AbstractVector) = KnnSorted(vec, one(Int32), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))

"""
    knnqueue(::Type{T}, k::Int) where {T<:AbstractKnn}

Creates a k-NN result queue of concrete type `T` (either [`KnnHeap`](@ref) or
[`KnnSorted`](@ref)) with capacity `k`, allocating a fresh backing vector of `k` zeroed
[`IdDist`](@ref) items.

# Examples

```julia
res = knnqueue(KnnSorted, 3)  # capacity k = 3, freshly allocated storage
```
"""
knnqueue(::Type{T}, k::Int) where {T<:AbstractKnn} = knnqueue(T, zeros(IdDist, k))

#const xknn = xknn
#end
