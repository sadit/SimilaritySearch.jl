# This file is a part of SimilaritySearch.jl

"""
    serializeindex(file, parent::String, index::SearchGraph, meta, options::Dict)

Saves a SearchGraph index optimizing for large indexes.
The adjancency list is always saved as `StaticAdjacencyList`, so it must changed after loading if needed.
"""
function serializeindex(file, parent::String, index::SearchGraph, meta, options::Dict)
    adj = StaticAdjacencyList(index.adj)
    I = copy(index; adj)
    file[joinpath(parent, "index")] = I
end

"""
    loadindex(filename::AbstractString, db=nothing; parent="/", staticgraph=false)
    restoreindex(file, parent::String, index::SearchGraph, meta, options::Dict; staticgraph=false)

Loads a SearchGraph index

- `staticgraph=false`. Determines if the index uses a static or a dynamic adjacency list.
"""
function restoreindex(file, parent::String, index::SearchGraph, meta, options::Dict; staticgraph=false)
    adj = staticgraph ? index.adj : AdjacencyList(index.adj)
    copy(index; adj)
end
