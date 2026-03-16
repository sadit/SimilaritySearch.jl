# This file is a part of SimilaritySearch.jl

export AdjDict32

"""
    struct AdjDict32

Structure to represent a very sparse graph
"""
struct AdjDict32 <: AbstractAdjList{UInt32}
    end_point::Dict{UInt32,Vector{UInt32}} # ending point of the i-th edge
    glock::Threads.ReentrantLock # global locks
end

#Base.eltype(::AdjDict32) = Pair{UInt32,Vector{UInt32}}
Base.eachindex(adj::AdjDict32) = keys(adj.end_point)

#function Base.iterate(adj::AdjDict32, state=nothing)
#    S = state === nothing ? iterate(adj.end_point) : iterate(adj.end_point, state)
#    S === nothing && return nothing
#    S
#end

#function AdjDict32(L::Dict{UInt32,Vector{UInt32}})
#    AdjDict32(L, UInt32[], Threads.SpinLock())
#end

#function AdjDict32(L::Vector{Vector{UInt32}})
#    AdjDict32(Dict(pairs(L)), UInt32[], Threads.SpinLock())
#end

function AdjDict32(n::Int)
    L = Dict{UInt32,Vector{UInt32}}()
    sizehint!(L, max(n, 4))
    AdjDict32(L, Threads.ReentrantLock())
end

function Base.resize!(adj::AdjDict32, n)
    lock(adj.glock) do 
        sizehint!(adj.end_point, n)
    end

    adj
end

AdjDict32(adj::AdjDict32) = AdjDict32(deepcopy(adj.end_point))
@inline Base.length(adj::AdjDict32) = length(adj.end_point)

@inline function packed_neighbors(adj::AdjDict32, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    get(adj.end_point, i, nothing)
end

Base.@propagate_inbounds @inline function neighbors_length(adj::AdjDict32, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    L = get(adj.end_point, i, nothing)
    L === nothing ? 0 : length(L)
end

function _add_edge!(adj::AdjDict32, from::UInt32, to::UInt32, isdirect::Bool)
    #from == to && return
    L = get(adj.end_point, from, nothing)
    if L === nothing
        L = adj.end_point[from] = UInt32[]
    end

    push!(L, pack_edge(UInt32(to), isdirect))
end

function _add!(adj::AdjDict32, from::UInt32, to)
    L = get(adj.end_point, from, nothing)
    if L === nothing
        L = adj.end_point[from] = UInt32[]
    end

    for i in to
        # from == i && continue
        push!(L, pack_edge(UInt32(i), true))
    end
end

function link_rev_edges!(adj::AdjDict32, from::Integer)
    from = UInt32(from)
    to = packed_neighbors(adj, from)

    for i in to
        i, isdirect = unpack_edge(i)
        isdirect && _add_edge!(adj, i, from, false)
    end
end

function link_rev_edges!(adj::AdjDict32, from::Integer, to::Integer)
    from = UInt32(from)
    lock(adj.glock) do
        for i in from:to
            link_rev_edges!(adj, i)
        end
    end
end

Base.@propagate_inbounds @inline function add!(adj::AdjDict32, n::Integer, N)
    n = convert(UInt32, n)
    lock(adj.glock) do
        _add!(adj, n, N)
    end

    adj
end

Base.@propagate_inbounds @inline function add!(adj::AdjDict32, other::AbstractAdjList)
    lock(adj.glock) do    
        S = Set{UInt32}()            
        for from in eachindex(other)
            from = convert(UInt32, from)
            N = packed_neighbors(other, from)
            N === nothing && continue
            L = get(adj.end_point, from, nothing)
            
            if L !== nothing
                empty!(S)
                L = adj.end_point[from]
                union!(S, L)
                for p in N
                    p ∈ S && continue
                    push!(L, p)
                    push!(S, p)
                end
            else
                adj.end_point[from] = collect(UInt32, N)
            end
        end
    end

    adj
end
