# This file is a part of SimilaritySearch.jl

export AdjList,
    neighbors, add_edge!, add_edges!, add_vertex!, neighbors_length

"""
    struct AdjList


Structure to represent a sparse graph
"""
struct AdjList{EndPointType} <: AbstractAdjList{EndPointType}
    end_point::Vector{Vector{EndPointType}} # ending point of the i-th edge
    empty_cent::Vector{EndPointType}  # empty list centinel for `neighbors` func
    locks::Vector{Threads.SpinLock} # adjancency list lock
    glock::Threads.SpinLock # global locks
end

Base.eltype(adj::AdjList{EndPointType}) where EndPointType = Vector{EndPointType}

function AdjList(lists::Vector{Vector{EndPointType}}) where EndPointType
    locks = [Threads.SpinLock() for _ in 1:length(lists)]
    AdjList{EndPointType}(lists, EndPointType[], locks, Threads.SpinLock())
end


function AdjList(::Type{EndPointType}, n::Int) where EndPointType
    lists = Vector{Vector{EndPointType}}(undef, n)
    AdjList(lists)
end

AdjList(t::Type{EndPointType}; n::Int=0) where EndPointType = AdjList(t, n::Int)

function Base.resize!(adj::AdjList, n)
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

AdjList(adj::AdjList) = AdjList(deepcopy(adj.end_point))
@inline Base.length(adj::AdjList) = length(adj.locks)

Base.@propagate_inbounds @inline function neighbors(adj::AdjList, i::Integer)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? (adj.end_point[i]) : (adj.empty_cent)
end

Base.@propagate_inbounds @inline function neighbors_length(adj::AdjList, i::Integer)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? length(adj.end_point[i]) : 0
end

Base.@propagate_inbounds @inline function add_edge!(adj::AdjList{EndPointType}, i::Integer, end_point, order=nothing) where EndPointType
    i == 0 && return adj
    @inbounds lock(adj.locks[i])
    # @info Int(i) => end_point, neighbors_length(adj, i)
    try
        if isassigned(adj.end_point, i)
            @inbounds list = adj.end_point[i]
            push!(list, end_point)
            order === nothing || sort_last_item!(order, list)
        else
            @inbounds adj.end_point[i] = EndPointType[end_point]
            sizehint!(adj.end_point[i], initial_size(adj))
        end
    finally
        @inbounds unlock(adj.locks[i])
    end

    adj
end

initial_size(adj::AdjList) = 8

Base.@propagate_inbounds @inline function add_edges!(adj::AdjList{EndPointType}, i::Integer, neighbors::Vector{EndPointType}) where EndPointType
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

Base.@propagate_inbounds @inline function add_edges!(adj::AdjList{EndPointType}, i::Integer, neighbors) where EndPointType
    i == 0 && return adj
    @inbounds lock(adj.locks[i])
    try
        if isassigned(adj.end_point, i)
            append!(adj.end_point[i], neighbors)
        else
            adj.end_point[i] = Vector(neighbors)
        end
    finally
        @inbounds unlock(adj.locks[i])
    end

    adj
end

Base.@propagate_inbounds @inline function add_vertex!(adj::AdjList{T}) where T
    l = T[]
    sizehint!(l, initial_size(adj))
    add_vertex!(adj, l)
end

Base.@propagate_inbounds @inline function add_vertex!(adj::AdjList{T}, neighbors) where T
    lock(adj.glock)
    try
        push!(adj.end_point, neighbors)
        push!(adj.locks, Threads.SpinLock())
    finally
        unlock(adj.glock)
    end

    neighbors
end

