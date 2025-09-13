# This file is a part of SimilaritySearch.jl

#module KnnResult

# export AbstractKnnQueueesult
export AbstractKnn, KnnHeap, KnnSorted, knnqueue, IdWeight
export push_item!, covradius, maxlength, reuse!, viewitems, sortitems!, pop_max!, nearest, frontier
export DistView, IdView

abstract type AbstractKnn
end

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

IdView(res::AbstractVector{IdWeight}) = (res[i].id for i in eachindex(res))
DistView(res::AbstractVector{IdWeight}) = (res[i].weight for i in eachindex(res))

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
