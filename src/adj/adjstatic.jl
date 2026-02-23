# This file is a part of SimilaritySearch.jl

export StaticAdjList

struct StaticAdjList{T} <: AbstractAdjList{T}
    offset::Vector{Int64}
    end_point::Vector{T}
end

Base.length(adj::StaticAdjList) = length(adj.offset)
Base.eltype(adj::StaticAdjList{T}) where T = Pair{T,typeof(view(adj.end_point, 1:1))}
Base.eachindex(adj::StaticAdjList) = eachindex(adj.offset)

function Base.iterate(adj::StaticAdjList{T}, i=1) where T
    i = T(i)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    i => neighbors(adj, i), i+1
end

function StaticAdjList(adj::StaticAdjList; offset=adj.offset, end_point=adj.end_point)
    StaticAdjList(offset, end_point)
end

function StaticAdjList(adj::AbstractAdjList{T}) where T
    n = length(adj)
    @show n
    offset = Vector{Int64}(undef, n)
    end_point = let N = sum(length(N) for (_, N) in adj)
        Vector{T}(undef, N)
    end

    i = 1
    s = 0
    @inbounds @inbounds for (j, N) in adj
        s += length(N)
        offset[j] = s

        for l in N
            end_point[i] = l
            i += 1
        end
    end

    StaticAdjList{T}(offset, end_point)
end

Base.@propagate_inbounds @inline function neighbors(adj::StaticAdjList, i::Integer)
    @inbounds sp::Int64 = i == 1 ? 1 : adj.offset[i-1] + 1
    @inbounds ep = adj.offset[i]
    view(adj.end_point, sp:ep)
end

Base.@propagate_inbounds @inline function neighbors_length(adj::StaticAdjList, i::Integer)
    @inbounds if i == 1
        adj.offset[i]
    else
        adj.offset[i] - adj.offset[i-1]
    end
end

function add!(adj::StaticAdjList, n, N)
    error("ERROR: unsupported add! on a static adjacent list")
end

function add!(adj::StaticAdjList, N)
    error("ERROR: unsupported add! on a static adjacent list")
end
