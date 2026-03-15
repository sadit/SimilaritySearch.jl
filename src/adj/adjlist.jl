# This file is a part of SimilaritySearch.jl

export AdjList32,
    packed_neighbors, neighbors_length, add!, unpack_edge, isdirect_edge

"""
    struct AdjList32

Structure to represent a sparse graph
"""
struct AdjList32 <: AbstractAdjList{UInt32}
    end_point::Vector{Vector{UInt32}} # ending point of the i-th edge
    glock::Threads.ReentrantLock # global locks
end

#Base.eltype(adj::AdjList32) = Pair{UInt32,Vector{UInt32}}
Base.eachindex(adj::AdjList32) = eachindex(adj.end_point)

#function Base.iterate(adj::AdjList32, i=1)
#    i = UInt32(i)
#    n = length(adj)
#    (n == 0 || i > n) && return nothing
#    i => packed_neighbors(adj, i), i+1
#end

#function AdjList32(A::Vector{Vector{UInt32}})
#    AdjList32(A, Threads.ReentrantLock())
#end

function AdjList32(n::Integer)
    AdjList32(Vector{Vector{UInt32}}(undef, n), Threads.ReentrantLock())
end

function Base.resize!(adj::AdjList32, n::Integer)
    lock(adj.glock) do
        resize!(adj.end_point, n)
    end

    adj
end

AdjList32(adj::AdjList32) = AdjList32(deepcopy(adj.end_point))
@inline Base.length(adj::AdjList32) = length(adj.end_point)

@inline pack_edge(i::UInt32, isdirect::Bool) = isdirect ? i : (i | 0x8000_0000)
@inline isdirect_edge(i::UInt32) = (i & 0x8000_0000) === 0x8000_0000
@inline unpack_edge(i::UInt32) = (i & 0x7fff_ffff, isdirect_edge(i))

Base.@propagate_inbounds @inline function packed_neighbors(adj::AdjList32, i::Integer)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? adj.end_point[i] : nothing
end

Base.@propagate_inbounds @inline function neighbors_length(adj::AdjList32, i::Integer)
    isassigned(adj.end_point, i) ? length(adj.end_point[i]) : 0
end

function _add_edge!(adj::AdjList32, from::UInt32, to::UInt32, isdirect::Bool)
    #from == to && return
    p = pack_edge(to, isdirect)

    if isassigned(adj.end_point, from)
        push!(adj.end_point[from], p)
    else
        adj.end_point[from] = UInt32[p]
    end
end

function _add_edge_list!(adj::AdjList32, from::UInt32, to)
    from > length(adj) && resize!(adj, from)

    if isassigned(adj.end_point, from)
        L = adj.end_point[from]
    else
        L = UInt32[]
        adj.end_point[from] = L
    end

    for i in to
        # from == i && continue
        push!(L, pack_edge(UInt32(i), true))
    end
end

Base.@propagate_inbounds @inline function add!(adj::AdjList32, from::Integer, to; linkrev::Bool=true)
    from = convert(UInt32, from)
    lock(adj.glock) do
        _add_edge_list!(adj, from, to)
        if linkrev
            let max = maximum(to, init=from)
                max > length(adj) && resize!(adj, max)
            end
            for i in to
                _add_edge!(adj, i, from, false)
            end
        end
    end

    adj
end

Base.@propagate_inbounds @inline function add!(adj::AdjList32, other::AbstractAdjList; linkrev::Bool=true)
    lock(adj.glock) do
        let n = max(length(other), length(adj))
            n > length(adj) && resize!(adj, n)
        end

        S = Set{UInt32}()
        for from in eachindex(other)
            from = convert(UInt32, from)
            N = packed_neighbors(other, from)
            N === nothing && continue
            if isassigned(adj.end_point, from)
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
