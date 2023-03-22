# This file is a part of SimilaritySearch.jl
module AdjacencyLists

abstract type AbstractAdjacencyList{EndPointType} end
export AbstractAdjacencyList, AdjacencyList, StaticAdjacencyList,
    neighbors, add_edge!, add_edges!, add_vertex!, neighbors_length,
    IdWeight, IdIntWeight, 
    sort_last_item!, IdOrder, WeightOrder, RevWeightOrder

using Base.Order
import Base.Order: lt

Base.eachindex(adj::AbstractAdjacencyList) = 1:length(adj)

function Base.iterate(adj::AbstractAdjacencyList, i::Int=1)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    @inbounds neighbors(adj, i), i+1
end

struct IdWeight
    id::UInt32
    weight::Float32
end

struct IdIntWeight
    id::UInt32
    weight::Int32
end

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
    item = plist[end]

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

struct AdjacencyList{EndPointType} <: AbstractAdjacencyList{EndPointType}
    end_point::Vector{Vector{EndPointType}}
    empty_cent::Vector{EndPointType}  # empty list centinel for `neighbors` func
    locks::Vector{Threads.SpinLock}
    glock::Threads.SpinLock
end

Base.eltype(adj::AdjacencyList{EndPointType}) where EndPointType = Vector{EndPointType}


function AdjacencyList(lists::Vector{Vector{EndPointType}}) where EndPointType
    locks = [Threads.SpinLock() for _ in 1:length(lists)]
    AdjacencyList{EndPointType}(lists, EndPointType[], locks, Threads.SpinLock())
end


function AdjacencyList(::Type{EndPointType}, n::Int) where EndPointType
    lists = Vector{Vector{EndPointType}}(undef, n)
    AdjacencyList(lists)
end

AdjacencyList(t::Type{EndPointType}; n::Int=0) where EndPointType = AdjacencyList(t, n::Int)

function Base.resize!(adj::AdjacencyList, n)
    lock(adj.glock)

    try
        len = length(adj.locks)
        resize!(adj.locks, n)
        @inbounds for i in len+1:n
            adj.locks[i] = Threads.SpinLock()
        end

        resize!(adj.end_point, n)
    finally
        unlock(adj.glock)
    end

    adj
end

AdjacencyList(adj::AdjacencyList) = AdjacencyList(deepcopy(adj.end_point))
@inline Base.length(adj::AdjacencyList) = length(adj.locks)

Base.@propagate_inbounds @inline function neighbors(adj::AdjacencyList, i::Integer)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? adj.end_point[i] : adj.empty_cent
end

Base.@propagate_inbounds @inline function neighbors_length(adj::AdjacencyList, i::Integer)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? length(adj.end_point[i]) : 0
end

Base.@propagate_inbounds @inline function add_edge!(adj::AdjacencyList{EndPointType}, i::Integer, end_point, order=nothing) where EndPointType
    i == 0 && return adj
    @inbounds lock(adj.locks[i])

    try
        if isassigned(adj.end_point, i)
            @inbounds list = adj.end_point[i]
            push!(list, end_point)
            order === nothing || sort_last_item!(order, list)
        else
            @inbounds adj.end_point[i] = EndPointType[end_point]
        end        
    finally
        @inbounds unlock(adj.locks[i])
    end

    adj
end

Base.@propagate_inbounds @inline function add_edges!(adj::AdjacencyList{EndPointType}, i::Integer, neighbors::Vector{EndPointType}) where EndPointType
    i == 0 && return adj
    @inbounds lock(adj.locks[i])
    try
        if isassigned(adj.end_point, i)
            append!(adj.end_point[i], neighbors)
        else
            adj.end_point[i] = neighbors
        end
    finally
        @inbounds unlock(adj.locks[i])
    end

    adj
end

Base.@propagate_inbounds @inline function add_edges!(adj::AdjacencyList{EndPointType}, i::Integer, neighbors) where EndPointType
    i == 0 && return adj
    @inbounds lock(adj.locks[i])
    try
        if !isassigned(adj.end_point, i)
            adj.end_point[i] = Vector(neighbors)
        else
            append!(adj.end_point[i], neighbors)
        end
    finally
        @inbounds unlock(adj.locks[i])
    end

    adj
end

Base.@propagate_inbounds @inline function add_vertex!(adj::AdjacencyList{T}) where T
    add_vertex!(adj, T[])
end

Base.@propagate_inbounds @inline function add_vertex!(adj::AdjacencyList{T}, neighbors) where T
    lock(adj.glock)
    try
        push!(adj.end_point, neighbors)
        push!(adj.locks, Threads.SpinLock())
    finally
        unlock(adj.glock)
    end

    neighbors
end


struct StaticAdjacencyList{EndPointType} <: AbstractAdjacencyList{EndPointType}
    offset::Vector{Int64}
    end_point::Vector{EndPointType}
end

Base.length(adj::StaticAdjacencyList) = length(adj.offset)
Base.eltype(adj::StaticAdjacencyList{EndPointType}) where EndPointType = typeof(view(adj.end_point, 1:1))

function StaticAdjacencyList(adj::StaticAdjacencyList; offset=adj.offset, end_point=adj.end_point)
    StaticAdjacencyList(offset, end_point)
end

Base.@propagate_inbounds @inline function neighbors(adj::StaticAdjacencyList, i::Integer)
    @inbounds sp::Int64 = i == 1 ? 1 : adj.offset[i-1] + 1
    @inbounds ep = adj.offset[i]
    view(adj.end_point, sp:ep)
end

Base.@propagate_inbounds @inline function neighbors_length(adj::StaticAdjacencyList, i::Integer)
    @inbounds sp::Int64 = i == 1 ? 1 : adj.offset[i-1] + 1
    @inbounds ep = adj.offset[i]
    length(ep - sp + 1)
end



function add_edge!(adj::StaticAdjacencyList, i::Integer, end_point)
    error("ERROR: unsupported add_edge! on a static adjacent list")
end

function add_edges!(adj::StaticAdjacencyList, i::Integer, neighbors)
    error("ERROR: unsupported add_edges! on a static adjacent list")
end

function add_vertex!(adj::StaticAdjacencyList)
    error("ERROR: unsupported add_vertext! on a static adjacent list")
end

function StaticAdjacencyList(adj::AdjacencyList{EndPointType}) where EndPointType
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

    StaticAdjacencyList{EndPointType}(offset, end_point)
end

function AdjacencyList(A::StaticAdjacencyList{EndPointType}) where EndPointType
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

    AdjacencyList(adj)
end

import SparseArrays: sparse

"""
    sparse(idx::AbstractAdjacencyList, val=1f0)

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
function sparse(adj::AbstractAdjacencyList{EndPointType}, val::AbstractFloat=1f0) where {EndPointType<:Integer}
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
    sparse(idx::AbstractAdjacencyList{IdWeight}) 
 
Creates an sparse matrix (from SparseArrays) from `idx`
"""
sparse(adj::AbstractAdjacencyList{IdWeight}) = sparse_from_adj(adj, UInt32, Float32)
sparse(adj::AbstractAdjacencyList{IdIntWeight}) = sparse_from_adj(adj, UInt32, Int32)

function sparse_from_adj(adj::AbstractAdjacencyList, IType, FType)
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
