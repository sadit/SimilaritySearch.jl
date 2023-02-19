# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::SearchGraph, meta, options::Dict)

Saves a SearchGraph index optimizing for large indexes.
The adjancency list is always saved as `StaticAdjacencyList`, so it must changed after loading if needed.
"""
function saveindex(filename::AbstractString, index::SearchGraph, meta, options::Dict)
    adj = StaticAdjacencyList(index.adj)
    I = copy(index; adj)
    jldsave(filename; index=I, meta, options)
end

"""
    loadindex(filename::AbstractString, db=nothing; staticgraph=false)
    restoreindex(index::SearchGraph, meta, options::Dict, f; staticgraph=false)

Loads a SearchGraph index

- `staticgraph=false`. Determines if the index uses a static or a dynamic adjacency list.
"""
function restoreindex(index::SearchGraph, meta, options::Dict, f; staticgraph=false)
    adj = staticgraph ? index.adj : AdjacencyList(index.adj)
    copy(index; adj), meta, options
end