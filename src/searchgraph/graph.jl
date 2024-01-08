# This file is a part of SimilaritySearch.jl

using Dates



### Basic operations on the index

"""
    struct SearchGraph <: AbstractSearchIndex

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `search_algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

- `hints`: Initial points for exploration (empty hints imply using random points)

Note: Parallel insertions should be made through `append!` or `index!` function with `parallel_block > 1`
"""
@with_kw struct SearchGraph{DistType<:SemiMetric, DataType<:AbstractDatabase, AdjType<:AbstractAdjacencyList, SType<:LocalSearchAlgorithm}<:AbstractSearchIndex
    dist::DistType = SqL2Distance()
    db::DataType = VectorDatabase()
    adj::AdjType = AdjacencyLists.AdjacencyList(UInt32)
    hints::Vector{Int32} = UInt32[]
    search_algo::SType = BeamSearch()
    len::Ref{Int64} = Ref(zero(Int64))
end

Base.copy(G::SearchGraph; 
    dist=G.dist,
    db=G.db,
    adj=G.adj,
    hints=G.hints,
    search_algo=copy(G.search_algo),
    len=Ref(length(G)),
) = SearchGraph(; dist, db, adj, hints, search_algo, len)

@inline Base.length(g::SearchGraph)::Int64 = g.len[]

include("beamsearch.jl")
## parameter optimization and neighborhood definitions
include("optbs.jl")
include("neighborhood.jl")
include("hints.jl")

"""
    search(index::SearchGraph, context::SearchGraphContext, q, res; hints=index.hints

Solves the specified query `res` for the query object `q`.
"""
function search(index::SearchGraph, context::SearchGraphContext, q, res::KnnResult; hints=index.hints)
    if length(index) > 0
        search(index.search_algo, index, context, q, res, hints)
    else
        SearchResult(res, 0)
    end
end
