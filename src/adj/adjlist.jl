# This file is a part of SimilaritySearch.jl

export AdjList,
    neighbors, neighbors_length, add!

"""
    struct AdjList

Structure to represent a sparse graph
"""
struct AdjList{T} <: AbstractAdjList{T}
    end_point::Vector{Vector{T}} # ending point of the i-th edge
    glock::Threads.ReentrantLock # global locks
end

Base.eltype(adj::AdjList{T}) where T = Pair{T,Vector{T}}
Base.eachindex(adj::AdjList) = eachindex(adj.end_point)

function Base.iterate(adj::AdjList{T}, i=1) where T
    i = T(i)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    i => neighbors(adj, i), i+1
end

function AdjList(A::Vector{Vector{T}}) where T
    AdjList{T}(A, Threads.ReentrantLock())
end

function AdjList(::Type{T}, n::Integer=0) where T
    AdjList(Vector{Vector{T}}(undef, n))
end

function Base.resize!(adj::AdjList, n::Integer)
    lock(adj.glock) do
        resize!(adj.end_point, n)
    end

    adj
end

AdjList(adj::AdjList) = AdjList(deepcopy(adj.end_point))
@inline Base.length(adj::AdjList) = length(adj.end_point)

Base.@propagate_inbounds @inline function neighbors(adj::AdjList, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? adj.end_point[i] : nothing
end

Base.@propagate_inbounds @inline function neighbors_length(adj::AdjList, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? length(adj.end_point[i]) : 0
end

Base.@propagate_inbounds @inline function add!(adj::AdjList{T}, n::Integer, N) where T
    lock(adj.glock) do
        n > length(adj) && resize!(adj, n)
        
        if isassigned(adj.end_point, n)
            append!(adj.end_point[n], N)
        else
            adj.end_point[n] = collect(T, N)
        end
    end

    adj
end

Base.@propagate_inbounds @inline function add!(adj::AdjList{T}, iter) where T
    n = max(length(iter), length(adj))
    lock(adj.glock) do
        n > length(adj) && resize!(adj, n)
        
        for (i, N) in iter
            add!(adj, i, N)
        end
    end

    adj
end
