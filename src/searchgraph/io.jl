# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::SearchGraph, meta::Dict)

Saves a SearchGraph index optimizing for large indexes.
The adjancency list is always saved as `StaticAdjacencyList`, so it must changed after loading if needed.
"""
function saveindex(filename::AbstractString, index::SearchGraph, meta::Dict)
    index.adj isa StaticAdjacencyList
    I = copy(index; adj=StaticAdjacencyList(index.adj))  # 
    jldsave(filename; index=I, meta)
end
