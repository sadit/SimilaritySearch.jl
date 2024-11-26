# This file is a part of SimilaritySearch.jl

using Dates

### Basic operations on the index

"""
    struct SearchGraph <: AbstractSearchIndex

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

- `hints`: Initial points for exploration (empty hints imply using random points)

Note: Parallel insertions should be made through `append!` or `index!` function with `parallel_block > 1`
"""
@with_kw struct SearchGraph{DIST<:SemiMetric,
                            DB<:AbstractDatabase,
                            ADJ<:AbstractAdjacencyList,
                            HINTS,
                            SEARCH<:LocalSearchAlgorithm
                           } <: AbstractSearchIndex
    dist::DIST = SqL2Distance()
    db::DB = VectorDatabase()
    adj::ADJ = AdjacencyLists.AdjacencyList(UInt32)
    hints::HINTS = UInt32[]
    algo::SEARCH = BeamSearch()
    len::Ref{Int64} = Ref(zero(Int64))
end

Base.copy(G::SearchGraph; 
    dist=G.dist,
    db=G.db,
    adj=G.adj,
    hints=G.hints,
    algo=copy(G.algo),
    len=Ref(length(G)),
) = SearchGraph(; dist, db, adj, hints, algo, len)

@inline Base.length(g::SearchGraph)::Int64 = g.len[]

"""
    enqueue_item!(index::SearchGraph, q, obj, res::KnnResult, objID, vstate)

Internal function that evaluates the distance between a database object `obj` with id `objID` and the query `q`.
It helps to evaluate, mark as visited, and enqueue in the result set.
"""
@inline function enqueue_item!(index::SearchGraph, q, obj, res::KnnResult, objID, vstate)::Int
    check_visited_and_visit!(vstate, convert(UInt64, objID)) && return 0
    d = evaluate(distance(index), q, database(index, objID))
    push_item!(res, objID, d)
    1
end

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
        search(index.algo, index, context, q, res, hints)
    else
        SearchResult(res, 0)
    end
end
