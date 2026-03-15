# This file is a part of SimilaritySearch.jl

#module KnnResult

# export AbstractKnnQueueesult
export AbstractKnn, KnnHeap, KnnSorted, knnqueue, IdDist
export push_item!, covradius, maxlength, reuse!, viewitems, sortitems!, pop_max!, nearest, frontier
export DistView, IdView
export distance_evaluations, block_evaluations

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

struct IdView{ARR}
    A::ARR
end

Base.eltype(::IdView) = UInt32
Base.length(res::IdView) = length(res.A)
Base.size(res::IdView) = size(res.A)
Base.firstindex(res::IdView) = 1
Base.lastindex(res::IdView) = length(res)
Base.eachindex(res::IdView) = firstindex(res):lastindex(res)
Base.getindex(res::IdView{<:AbstractMatrix{IdDist}}, i...) = res.A[i...].id
Base.getindex(res::IdView{<:AbstractVector{IdDist}}, i::Integer) = res.A[i].id
Base.getindex(res::IdView{<:AbstractVector{<:Integer}}, i::Integer) = res.A[i]
Base.getindex(res::IdView{<:KnnHeap}, i::Integer) = res.A.items[i].id
Base.getindex(res::IdView{<:KnnSorted}, i::Integer) = res.A.items[res.A.sp+i-1].id

struct DistView{ARR}
    A::ARR
end

Base.eltype(::DistView) = Float32
Base.length(res::DistView) = length(res.A)
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
    knnqueue(::{KnnHeap,KnnSorted}, vec::AbstractVector)
    knnqueue(::{KnnHeap,KnnSorted}, ksearch::Int)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push_item!`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
knnqueue(::Type{KnnHeap}, vec::AbstractVector) = KnnHeap(vec, zero(IdDist), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
knnqueue(::Type{KnnSorted}, vec::AbstractVector) = KnnSorted(vec, one(Int32), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
knnqueue(::Type{T}, k::Int) where {T<:AbstractKnn} = knnqueue(T, zeros(IdDist, k))

#const xknn = xknn
#end
