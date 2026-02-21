# This file is a part of SimilaritySearch.jl

export StaticAdjList

struct StaticAdjList{EndPointType} <: AbstractAdjList{EndPointType}
    offset::Vector{Int64}
    end_point::Vector{EndPointType}
end

Base.length(adj::StaticAdjList) = length(adj.offset)
Base.eltype(adj::StaticAdjList{EndPointType}) where EndPointType = typeof(view(adj.end_point, 1:1))

function StaticAdjList(adj::StaticAdjList; offset=adj.offset, end_point=adj.end_point)
    StaticAdjList(offset, end_point)
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

function add_edge!(adj::StaticAdjList, i::Integer, end_point)
    error("ERROR: unsupported add_edge! on a static adjacent list")
end

function add_edges!(adj::StaticAdjList, i::Integer, neighbors)
    error("ERROR: unsupported add_edges! on a static adjacent list")
end

function add_vertex!(adj::StaticAdjList)
    error("ERROR: unsupported add_vertext! on a static adjacent list")
end
