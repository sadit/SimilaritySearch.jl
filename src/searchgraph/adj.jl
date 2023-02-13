# This file is a part of SimilaritySearch.jl

abstract type AbstractAdjacencyList end

Base.eachindex(adj::AbstractAdjacencyList) = 1:length(adj)

@with_kw struct AdjacencyList <: AbstractAdjacencyList
    links::Vector{Vector{UInt32}} = Vector{Vector{UInt32}}(undef, 0)
end

AdjacencyList(adj::AdjacencyList) = AdjacencyList(deepcopy(adj))

Base.@propagate_inbounds function neighbors(adj::AdjacencyList, i::Integer)
    adj.links[i]
end

Base.@propagate_inbounds function add_edge!(adj::AdjacencyList, i::Integer, neighbor::Integer, dist::Float32)
    push!(adj.links[i], convert(UInt32, neighbor))
end

Base.@propagate_inbounds function add_vertex!(adj::AdjacencyList)
    add_vertex!(adj, UInt32[])
end

Base.@propagate_inbounds function add_vertex!(adj::AdjacencyList, neighbors::Vector{UInt32})
    push!(adj.links, neighbors)
    neighbors
end

struct StaticAdjacencyList <: AbstractAdjacencyList
    offset::Vector{Int64}
    links::Vector{UInt32}
end

Base.length(adj::StaticAdjacencyList) = length(adj.offset)

function StaticAdjacencyList(adj::StaticAdjacencyList; offset=adj.offset, links=adj.links)
    StaticAdjacencyList(offset, links)
end

Base.@propagate_inbounds function neighbors(adj::StaticAdjacencyList, i::Integer)
    sp::Int64 = i == 1 ? 1 : adj.links[i-1]
    ep = adj.links[i]
    view(adj.links, sp:ep)
end

function add_edge!(adj::StaticAdjacencyList, i::Integer, neighbor::Integer, dist::Float32)
    error("ERROR: unsupported add_edge! on a static adjacent list")
end

function add_vertex!(adj::StaticAdjacencyList)
    error("ERROR: unsupported add_vertext! on a static adjacent list")
end

function StaticAdjacencyList(adj::AdjacencyList)
    n = length(adj)
    offset = Vector{Int64}(undef, n)
    links = Vector{UInt32}(undef, n)

    i = 1
    @inbounds for (j, L) in enumerate(adj)
        s += length(L)
        offset[j] = s

        for l in L
            links[i] = l
            i += 1
        end
    end

    StaticAdjacencyList(offset, links)
end

function AdjacencyList(A::StaticAdjacencyList)
    n = length(A)
    adj = Vector{Vector{UInt32}}(undef, n)

    for objID in 1:n
        C = A[objID]
        len = length(C)
        children = Vector{UInt32}(undef, len)

        for i in 1:len
            children[i] = C[i]
        end

        adj[objID] = children
    end

    AdjacencyList(adj)
end
