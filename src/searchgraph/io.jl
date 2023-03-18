# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::SearchGraph, meta, options::Dict)

Saves a SearchGraph index optimizing for large indexes.
The adjancency list is always saved as `StaticAdjacencyList`, so it must changed after loading if needed.
"""
function saveindex(file::JLD2.JLDFile, index::SearchGraph, meta, options::Dict; parent="/")
    adj = StaticAdjacencyList(index.adj)
    I = copy(index; adj)
    file[joinpath(parent, "options")] = options 
    file[joinpath(parent, "meta")] = meta
    file[joinpath(parent, "index")] = I
end

"""
    loadindex(filename::AbstractString, db=nothing; staticgraph=false)
    restoreindex(index::SearchGraph, meta, options::Dict; staticgraph=false)

Loads a SearchGraph index

- `staticgraph=false`. Determines if the index uses a static or a dynamic adjacency list.
"""
function restoreindex(index::SearchGraph, meta, options::Dict; staticgraph=false)
    adj = staticgraph ? index.adj : AdjacencyList(index.adj)
    copy(index; adj)
end
