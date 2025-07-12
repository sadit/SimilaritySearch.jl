# This file is a part of SimilaritySearch.jl

#module KnnResult

# export AbstractResult
export AbstractKnn, Knn, knn, XKnn, xknn, IdWeight
export push_item!, covradius, maxlength, reuse!, viewitems, sortitems!, DistView, IdView

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
include("knn.jl")
include("xknn.jl")

IdView(res::AbstractVector{IdWeight}) = (res[i].id for i in eachindex(res))
DistView(res::AbstractVector{IdWeight}) = (res[i].weight for i in eachindex(res))

"""
    knn(ksearch)
    knn(vec)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push_item!`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
knn(vec::AbstractVector) = Knn(vec, IdWeight(Int32(0), 0f0), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
knn(k::Int) = knn(Vector{IdWeight}(undef, k))


"""
    Xknn(ksearch)
    Xknn(vec)

Creates a priority queue with fixed capacity (`ksearch`) representing a Xknn result set.
It starts with zero items and grows with [`push_item!`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
xknn(vec::AbstractVector) = XKnn(vec, one(Int32), zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
xknn(k::Int) = xknn(Vector{IdWeight}(undef, k))

#const xknn = xknn
#end
