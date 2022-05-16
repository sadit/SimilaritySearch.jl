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
    @with_kw mutable struct Neighborhood
    
Determines the size of the neighborhood, \$k\$ is adjusted as a callback, and it is intended to affect previously inserted vertices.
The neighborhood is designed to consider two components \$k=in+out\$, i.e. _in_coming and _out_going edges for each vertex.
- The \$out\$ size is computed as \$minsize + \\log(logbase, n)\$ where \$n\$ is the current number of indexed elements; this is computed searching
for \$out\$  elements in the current index.
- The \$in\$ size is unbounded.
- reduce is intended to postprocess neighbors (after search process, i.e., once out edges are computed); do not change \$k\$ but always must return a copy of the reduced result set.

Note: Set \$logbase=Inf\$ to obtain a fixed number of \$in\$ nodes; and set \$minsize=0\$ to obtain a pure logarithmic growing neighborhood.

"""
@with_kw mutable struct Neighborhood
    ksearch::Int32 = 2
    logbase::Float32 = 2
    minsize::Int32 = 2
    reduce::NeighborhoodReduction = SatNeighborhood()
end

Base.copy(N::Neighborhood; ksearch=N.ksearch, logbase=N.logbase, minsize=N.minsize, reduce=copy(N.reduce)) =
    Neighborhood(; ksearch, logbase, minsize, reduce)

struct NeighborhoodSize <: Callback end

"""
    struct SearchGraph <: AbstractSearchContext

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `search_algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

- `hints`: Initial points for exploration (empty hints imply using random points)

Note: Parallel insertions should be made through `append!` or `index!` function with `parallel_block > 1`

"""
@with_kw struct SearchGraph{DistType<:SemiMetric, DataType<:AbstractDatabase, SType<:LocalSearchAlgorithm}<:AbstractSearchContext
    dist::DistType = SqL2Distance()
    db::DataType = VectorDatabase()
    links::Vector{Vector{Int32}} = Vector{Int32}[]
    locks::Vector{Threads.SpinLock} = Threads.SpinLock[]
    hints::Vector{Int32} = Int32[]
    search_algo::SType = BeamSearch()
    neighborhood::Neighborhood = Neighborhood()
    verbose::Bool = true
end

@with_kw struct SearchGraphCallbacks
    hints::Union{Nothing,Callback} = DisjointHints()
    neighborhood::Union{Nothing,Callback} = NeighborhoodSize()
    hyperparameters::Union{Nothing,Callback} = OptimizeParameters(kind=ParetoRecall())
    logbase::Float32 = 1.5
    starting::Int32 = 8
end

@inline Base.length(g::SearchGraph) = length(g.locks)
include("visitedvertices.jl")

Base.copy(g::SearchGraph;
        dist=g.dist,
        db=g.db,
        links=g.links,
        locks=g.locks,
        hints=g.hints,
        search_algo=copy(g.search_algo),
        neighborhood=copy(g.neighborhood),
        verbose=true
) = SearchGraph(; dist, db, links, locks, hints, search_algo, neighborhood, verbose)

## search algorithms

"""
    SearchGraphPools(results=GlobalKnnResult, vstates=GlobalVisitedVertices, beams=GlobalBeamKnnResult)

A set of pools to alleviate memory allocations in `SearchGraph` construction and searching. Relevant on multithreading scenarious where distance functions, `evaluate`
can call other metric indexes that can use these shared resources (globally defined).

Each pool is a vector of `Threads.nthreads()` preallocated objects of the required type.
"""
struct SearchGraphPools{VisitedVerticesType}
    beams::Vector{KnnResultShift}
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

"""
    execute_callbacks(index::SearchGraph, n=length(index), m=n+1)

Process all registered callbacks
"""
function execute_callbacks(callbacks::SearchGraphCallbacks, index::SearchGraph, n=length(index), m=n+1; force=false)
    if force || (n >= callbacks.starting && ceil(Int, log(callbacks.logbase, n)) != ceil(Int, log(callbacks.logbase, m)))
        callbacks.hints !== nothing && execute_callback(callbacks.hints, index)
        callbacks.neighborhood !== nothing && execute_callback(callbacks.neighborhood, index)
        callbacks.hyperparameters !== nothing && execute_callback(callbacks.hyperparameters, index)
    end
end