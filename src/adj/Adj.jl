# This file is a part of SimilaritySearch.jl
module Adj

abstract type AbstractAdjList{EndPointType} end

export AbstractAdjList, sort_last_item!, IdWeight, IdIntWeight,
    IdOrder, WeightOrder, RevWeightOrder

using Base.Order
import Base.Order: lt

Base.eachindex(adj::AbstractAdjList) = 1:length(adj)

function Base.iterate(adj::AbstractAdjList, i::Int=1)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    @inbounds neighbors(adj, i), i+1
end

"""
    IdWeight(id, weight)

Stores a pair of objects to be accessed. It is used in several places but mostly as an item in `KnnResult` algorithms where `weight` field is a distance instead of a weight
    
"""
struct IdWeight
    id::UInt32
    weight::Float32
end


"""
    IdIntWeight(id, weight)

Stores a pair of objects to be accessed. Similar to [`IdWeight`](@ref) but it stores an integer weight 
"""
struct IdIntWeight
    id::UInt32
    weight::Int32
end

Base.zero(::Type{IdWeight}) = IdWeight(zero(UInt32), zero(Float32))
Base.zero(::Type{IdIntWeight}) = IdWeight(zero(UInt32), zero(Int32))

struct IdOrderingType <: Ordering end
struct WeightOrderingType <: Ordering end
struct RevWeightOrderingType <: Ordering end
const IdOrder = IdOrderingType()
const WeightOrder = WeightOrderingType()
const RevWeightOrder = RevWeightOrderingType()

@inline lt(::IdOrderingType, a, b) = a.id < b.id
@inline lt(::WeightOrderingType, a, b) = a.weight < b.weight
@inline lt(::RevWeightOrderingType, a, b) = b.weight < a.weight
@inline lt(::IdOrderingType, a::Number, b::Number) = a < b
@inline lt(::WeightOrderingType, a::Number, b::Number) = a < b
@inline lt(::RevWeightOrderingType, a::Number, b::Number) = b < a

"""
    sort_last_item!(order::Ordering, plist)

Sorts the last push in place. It implements insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected to be really near of its sorted position)
"""
function sort_last_item!(order::Ordering, plist::AbstractVector)
    sp = 1
    pos = N = lastindex(plist)
    @inbounds item = plist[N]

    @inbounds while pos > sp && lt(order, item, plist[pos-1])
        pos -= 1
    end

    @inbounds if pos < N
        while N > pos
            plist[N] = plist[N-1]
            N -= 1
        end

        plist[N] = item
    end

    nothing
end

include("adjlist.jl")
include("adjstatic.jl")

function StaticAdjList(adj::AdjList{EndPointType}) where EndPointType
    n = length(adj)
    offset = Vector{Int64}(undef, n)
    N = sum(length(neighbors(adj, j)) for j in eachindex(adj))
    end_point = Vector{EndPointType}(undef, N)

    i = 1
    s = 0
    @inbounds @inbounds for j in eachindex(adj)
        L = neighbors(adj, j)
        s += length(L)
        offset[j] = s

        for l in L
            end_point[i] = l
            i += 1
        end
    end

    StaticAdjList{EndPointType}(offset, end_point)
end

function AdjList(A::StaticAdjList{EndPointType}) where EndPointType
    n = length(A)
    adj = Vector{Vector{EndPointType}}(undef, n)

    @inbounds for objID in 1:n
        C = neighbors(A, objID)
        len = length(C)
        lst = Vector{EndPointType}(undef, len)

        for i in 1:len
            lst[i] = C[i]
        end

        adj[objID] = lst
    end

    AdjList(adj)
end

import SparseArrays: sparse

"""
    sparse(idx::AbstractAdjList, val=1f0)

Creates an sparse matrix (from SparseArrays) from `idx` using `val` as value.

```
   I  
   ↓    1 2 3 4 5 … n  ← J
 L[1] = 0 1 0 0 1 … 0
 L[2] = 1 0 0 1 0 … 1
 L[3] = 1 0 1 0 0 … 1
 ⋮
 L[m] = 0 0 1 1 0 … 0
```
"""
function sparse(adj::AbstractAdjList{EndPointType}, val::AbstractFloat=1f0) where {EndPointType<:Integer}
    n = length(adj)
    I = EndPointType[]
    J = EndPointType[]
    F = eltype(val)[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(adj)
        L = neighbors(adj, i)
        for j in L
            push!(I, i)
            push!(J, j)
            push!(F, val)
        end
    end

    sparse(I, J, F, length(adj), n)
end

"""
    sparse(idx::AbstractAdjList{IdWeight}) 
 
Creates an sparse matrix (from SparseArrays) from `idx`
"""
sparse(adj::AbstractAdjList{IdWeight}) = sparse_from_adj(adj, UInt32, Float32)
sparse(adj::AbstractAdjList{IdIntWeight}) = sparse_from_adj(adj, UInt32, Int32)

function sparse_from_adj(adj::AbstractAdjList, IType, FType)
    n = length(adj)
    I = IType[]
    J = JType[]
    F = FType[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(adj)
        L = neighbors(adj, i)

        for s in L
            push!(I, i)
            push!(J, s.id)
            push!(F, s.weight)
        end
    end

    sparse(I, J, F, length(adj), n)
end

end