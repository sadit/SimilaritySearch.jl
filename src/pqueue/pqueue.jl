# This file is a part of SimilaritySearch.jl

#module KnnResult

# export AbstractKnnQueueesult
export AbstractKnn, KnnHeap, KnnSorted, knnqueue, IdWeight
export push_item!, covradius, maxlength, reuse!, viewitems, sortitems!, pop_max!, nearest, frontier
export DistView, IdView
export distance_evaluations, block_evaluations

abstract type AbstractKnn end

#=struct IdWeight
    id::UInt32
    weight::Float32
end=#

using Base.Order
import Base.Order: lt

struct WeightOrderingType <: Ordering end
struct RevWeightOrderingType <: Ordering end
const WeightOrder = WeightOrderingType()
const RevWeightOrder = RevWeightOrderingType()

@inline lt(::WeightOrderingType, a::IdWeight, b::IdWeight) = a.weight < b.weight
@inline lt(::RevWeightOrderingType, a::IdWeight, b::IdWeight) = b.weight < a.weight
@inline lt(::WeightOrderingType, a::Number, b::Number) = a < b
@inline lt(::RevWeightOrderingType, a::Number, b::Number) = b < a

include("heap.jl")
include("knnheap.jl")
include("knnsorted.jl")

@inline covradius(res::AbstractKnn)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)
@inline Base.maximum(res::AbstractKnn) = frontier(res).weight
@inline Base.argmax(res::AbstractKnn) = frontier(res).id
@inline Base.minimum(res::AbstractKnn) = nearest(res).weight
@inline Base.argmin(res::AbstractKnn) = nearest(res).id

Base.convert(::Type{T}, v::IdWeight) where {T<:Integer} = convert(T, v.id)
Base.convert(::Type{T}, v::IdWeight) where {T<:AbstractFloat} = convert(T, v.weight)
Base.convert(::Type{T}, v::AbstractVector{IdWeight}) where {T<:Vector{<:Integer}} = T(IdView(v))
Base.convert(::Type{T}, v::AbstractVector{IdWeight}) where {T<:Vector{<:AbstractFloat}} = T(DistView(v))
function Base.convert(::Type{T}, v::AbstractMatrix{IdWeight}) where {T<:Matrix{<:Integer}}
    X = T(undef, size(v))
    V = IdView(v)
    for i in eachindex(X)
        X[i] = V[i]
    end
    X
end

function Base.convert(::Type{T}, v::AbstractMatrix{IdWeight}) where {T<:Matrix{<:AbstractFloat}}
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

Base.length(res::IdView) = length(res.A)
Base.size(res::IdView) = size(res.A)
Base.firstindex(res::IdView) = 1
Base.lastindex(res::IdView) = length(res)
Base.eachindex(res::IdView) = firstindex(res):lastindex(res)
Base.getindex(res::IdView{<:AbstractMatrix{IdWeight}}, i...) = res.A[i...].id
Base.getindex(res::IdView{<:AbstractVector{IdWeight}}, i::Integer) = res.A[i].id
Base.getindex(res::IdView{<:AbstractVector{<:Integer}}, i::Integer) = res.A[i]
Base.getindex(res::IdView{<:KnnHeap}, i::Integer) = res.A.items[i].id
Base.getindex(res::IdView{<:KnnSorted}, i::Integer) = res.A.items[res.A.sp+i-1].id

struct DistView{ARR}
    A::ARR
end

Base.length(res::DistView) = length(res.A)
Base.size(res::DistView) = size(res.A)
Base.firstindex(res::DistView) = 1
Base.lastindex(res::DistView) = length(res)
Base.eachindex(res::DistView) = firstindex(res):lastindex(res)
Base.getindex(res::DistView{<:AbstractMatrix{IdWeight}}, i...) = res.A[i...].weight
Base.getindex(res::DistView{<:AbstractVector{IdWeight}}, i::Integer) = res.A[i].weight
Base.getindex(res::DistView{<:AbstractVector{<:AbstractFloat}}, i::Integer) = Float32(res.A[i])
Base.getindex(res::DistView{<:KnnHeap}, i::Integer) = res.A.items[i].weight
Base.getindex(res::DistView{<:KnnSorted}, i::Integer) = res.A.items[res.A.sp+i-1].weight

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
knnqueue(::Type{KnnHeap}, vec::AbstractVector) = KnnHeap(vec, zero(IdWeight), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
knnqueue(::Type{KnnSorted}, vec::AbstractVector) = KnnSorted(vec, one(Int32), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
knnqueue(::Type{T}, k::Int) where {T<:AbstractKnn} = knnqueue(T, zeros(IdWeight, k))

#const xknn = xknn
#end
