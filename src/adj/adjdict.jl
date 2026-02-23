# This file is a part of SimilaritySearch.jl

export AdjDict

"""
    struct AdjDict

Structure to represent a very sparse graph
"""
struct AdjDict{T} <: AbstractAdjList{T}
    end_point::Dict{T,Vector{T}} # ending point of the i-th edge
    empty_cent::Vector{T}  # empty list centinel for `neighbors` func
    glock::Threads.SpinLock # global locks
end

Base.eltype(::AdjDict{T}) where T = Pair{T,Vector{T}}
Base.eachindex(adj::AdjDict) = keys(adj.end_point)

function Base.iterate(adj::AdjDict{T}, state=nothing) where T
    S = state === nothing ? iterate(adj.end_point) : iterate(adj.end_point, state)
    S === nothing && return nothing
    S
end

function AdjDict(L::Dict{T,Vector{T}}) where T
    AdjDict{T}(L, T[], Threads.SpinLock())
end

function AdjDict(L::Vector{Vector{T}}) where T
    AdjDict{T}(Dict(pairs(L)), T[], Threads.SpinLock())
end

function AdjDict(::Type{T}, n::Int) where T
    L = Dict{T,Vector{T}}()
    sizehint!(L, n)
    AdjDict(L)
end

AdjDict(::Type{T}; n::Int=0) where T = AdjDict(T, n::Int)

function Base.resize!(adj::AdjDict, n)
    # do nothing
end

AdjDict(adj::AdjDict) = AdjDict(deepcopy(adj.end_point))
@inline Base.length(adj::AdjDict) = length(adj.end_point)

Base.@propagate_inbounds @inline function neighbors(adj::AdjDict, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    L = get(adj.end_point, i, nothing)
    L === nothing ? (adj.empty_cent) : L
end

Base.@propagate_inbounds @inline function neighbors_length(adj::AdjDict, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    L = get(adj.end_point, i, nothing)
    L === nothing ? 0 : length(L)
end

Base.@propagate_inbounds @inline function add!(adj::AdjDict{T}, n, N) where T
    lock(adj.glock) do
        L = get(adj.end_point, n, nothing)
        if L === nothing
            adj.end_point[n] = collect(T, N)
        else
            append!(L, N)
        end
    end

    adj
end

Base.@propagate_inbounds @inline function add!(adj::AdjDict{T}, iter) where T
    lock(adj.glock) do
        for (n, N) in iter
            L = get(adj.end_point, n, nothing)
            if L === nothing
                adj.end_point[n] = collect(T, N)
            else
                append!(L, N)
            end
        end
    end

    adj
end

