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

include("visitedvertices.jl")

## search algorithms

"""
    SearchGraphPools(results=GlobalKnnResult, vstates=GlobalVisitedVertices, beams=GlobalBeamKnnResult)

A set of pools to alleviate memory allocations in `SearchGraph` construction and searching. Relevant on multithreading scenarious where distance functions, `evaluate`
can call other metric indexes that can use these shared resources (globally defined).

Each pool is a vector of `Threads.nthreads()` preallocated objects of the required type.
"""
struct SearchGraphPools{VisitedVerticesType}
    beams::Vector{KnnResult}
    satnears::Vector{KnnResult}
    vstates::VisitedVerticesType
end

@inline function getvstate(len, pools::SearchGraphPools)
    @inbounds reuse!(pools.vstates[Threads.threadid()], len)
end

@inline function getbeam(bsize::Integer, pools::SearchGraphPools)
    @inbounds reuse!(pools.beams[Threads.threadid()], bsize)
end

@inline function getsatknnresult(pools::SearchGraphPools)
    reuse!(pools.satnears[Threads.threadid()], 1)
end

"""
    getpools(index::SearchGraph)

Creates or retrieve caches for the search graph.
"""
getpools(::SearchGraph; beams=GlobalBeamKnnResult, satnears=GlobalSatKnnResult, vstates=GlobalVisitedVertices) = SearchGraphPools(beams, satnears, vstates)

include("beamsearch.jl")
## parameter optimization and neighborhood definitions
include("optbs.jl")
include("neighborhood.jl")
include("hints.jl")

"""
    search(index::SearchGraph, q, res; hints=index.hints, pools=getpools(index))

Solves the specified query `res` for the query object `q`.
"""
function search(index::SearchGraph, q, res::KnnResult; hints=index.hints, pools=getpools(index))
    if length(index) > 0
        search(index.search_algo, index, q, res, hints, pools)
    else
        SearchResult(res, 0)
    end
end
