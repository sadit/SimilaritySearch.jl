# This file is a part of SimilaritySearch.jl

using Dates

"""
    abstract type NeighborhoodReduction end
    
Overrides `Base.reduce(::NeighborhoodReduction, res::KnnResult, index::SearchGraph)` to postprocess `res` using some criteria.
Called from `find_neighborhood`, and returns a new KnnResult struct (perhaps a copy of res) since `push_neighborhood` captures
the reference of its output.
"""
abstract type NeighborhoodReduction end
abstract type LocalSearchAlgorithm end

"""
    abstract type Callback end

Abstract type to trigger callbacks after some number of insertions.
SearchGraph stores the callbacks in `callbacks` (a dictionary that associates symbols and callback objects);
A SearchGraph object controls when callbacks are fired using `callback_logbase` and `callback_starting`

"""
abstract type Callback end

### Basic operations on the index

"""
    struct SearchGraph <: AbstractSearchIndex

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `search_algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

- `hints`: Initial points for exploration (empty hints imply using random points)

Note: Parallel insertions should be made through `append!` or `index!` function with `parallel_block > 1`

"""
@with_kw struct SearchGraph{DistType<:SemiMetric, DataType<:AbstractDatabase, SType<:LocalSearchAlgorithm}<:AbstractSearchIndex
    dist::DistType = SqL2Distance()
    db::DataType = VectorDatabase()
    links::Vector{Vector{Int32}} = Vector{Int32}[]
    locks::Vector{Threads.SpinLock} = Threads.SpinLock[]
    hints::Vector{Int32} = Int32[]
    search_algo::SType = BeamSearch()
    verbose::Bool = true
end

@inline Base.length(g::SearchGraph) = length(g.locks)
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
    @inbounds v = pools.vstates[Threads.threadid()]
    _init_vv(v, len)
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
        (res=res, cost=0)
    end
end
