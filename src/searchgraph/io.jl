# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::SearchGraph, meta::Dict)

Saves a SearchGraph index optimizing for large indexes.
The adjancency list is always saved as `StaticAdjacencyList`, so it must changed after loading if needed.
"""
function saveindex(filename::AbstractString, index::SearchGraph, meta::Dict)
    adj = StaticAdjacencyList(index.adj)
    I = copy(index; adj)
    jldsave(filename; index=I, meta)
end

"""
    loadindex(filename::AbstractString, db=nothing; staticgraph=false)
    restoreindex(index::SearchGraph, meta::Dict, f; staticgraph=false)

Loads a SearchGraph index

- `staticgraph=false`. Determines if the index uses a static or a dynamic adjacency list.
"""
function restoreindex(index::SearchGraph, meta::Dict, f; staticgraph=false)
    if staticgraph
        index, meta
    else
        copy(index; adj=AdjacencyList(index.adj)), meta
    end
    # index, meta
end