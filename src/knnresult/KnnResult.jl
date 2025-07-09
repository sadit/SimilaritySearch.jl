# This file is a part of SimilaritySearch.jl

#module KnnResult

# export AbstractResult
export AbstractKnn, Knn, knn, XKnn, xknn, IdWeight
export knnset, xknnset, KnnSet, knnpool, xknnpool, KnnPool
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

struct KnnSet{KNN}
    matrix::Matrix{IdWeight}
    knns::KNN
end

Base.length(s::KnnSet) = size(s.matrix, 2)

knnset(matrix::Matrix) = KnnSet(matrix, [knn(c) for c in eachcol(matrix)])
knnset(k::Integer, n::Integer) = knnset(zeros(IdWeight, k, n))

xknnset(matrix::Matrix) = KnnSet(matrix, [xknn(c) for c in eachcol(matrix)])
xknnset(k::Integer, n::Integer) = xknnset(zeros(IdWeight, k, n))

function reuse!(set::KnnSet, i::Integer, k::Integer=0)
    r = set.knns[i]
    reuse!(r, k == 0 ? r.maxlen : k)
end


struct KnnPool{KNN}
    matrix::Matrix{IdWeight}
    knns::KNN
end

Base.length(s::KnnPool) = size(s.matrix, 2)

knnpool(matrix::Matrix; poolsize::Int=Threads.nthreads()) = KnnPool(matrix, [knn(view(matrix, :, i)) for i in 1:poolsize])
knnpool(k::Integer, n::Integer; poolsize::Int=Threads.nthreads()) = knnpool(zeros(IdWeight, k, n); poolsize)

xknnpool(matrix::Matrix; poolsize::Int=Threads.nthreads()) = KnnPool(matrix, [xknn(view(matrix, :, i)) for i in 1:poolsize])
xknnpool(k::Integer, n::Integer; poolsize::Int=Threads.nthreads()) = xknnpool(zeros(IdWeight, k, n); poolsize)

function reuse!(pool::KnnPool, i::Integer, k::Integer=size(pool.matrix, 1))
    r = pool.knns[Threads.threadid()]
    r.items = view(pool.matrix, :, i)
    reuse!(r, k)
end

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
xknn(vec::AbstractVector) = XKnn(vec, zero(Int32), Int32(length(vec)), zero(Int32), zero(Int32))
xknn(k::Int) = xknn(Vector{IdWeight}(undef, k))

#const xknn = xknn
#end
