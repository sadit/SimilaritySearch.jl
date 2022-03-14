# This file is a part of SimilaritySearch.jl

using Dates
export LocalSearchAlgorithm, SearchGraph, SearchGraphPools, SearchGraphCallbacks, VisitedVertices, NeighborhoodReduction, index!, push_item!
export Callback

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
    results::Vector{KnnResult}
    beams::Vector{KnnResultShift}
    satnears::Vector{KnnResult}
    vstates::VisitedVerticesType
end

@inline function getknnresult(k::Integer, pools::SearchGraphPools)
    res = @inbounds pools.results[Threads.threadid()]
    reuse!(res, k)
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

getpools(::SearchGraph; results=GlobalKnnResult, beams=GlobalBeamKnnResult, satnears=GlobalSatKnnResult, vstates=GlobalVisitedVertices) = SearchGraphPools(results, beams, satnears, vstates)

include("beamsearch.jl")
## parameter optimization and neighborhood definitions
include("opt.jl")
include("neighborhood.jl")
include("hints.jl")

"""
    append!(
        index::SearchGraph,
        db;
        parallel_block=1,
        parallel_minimum_first_block=parallel_block,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )

Appends all items in db to the index. It can be made in parallel or sequentially.

Arguments:

- `index`: the search graph index
- `db`: the collection of objects to insert, an `AbstractDatabase` is the canonical input, but supports any iterable objects
- `parallel_block`: The number of elements that the multithreading algorithm process at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `parallel_minimum_first_block`: The number of sequential appends before running parallel.
Note: Parallel doesn't trigger callbacks inside blocks.
- `callbacks`: A `SearchGraphCallbacks` object to be called after some insertions
    (specified by the `callbacks` object). These callbacks are used to maintain the algorithm
    in good shape after many insertions (adjust hyperparameters and the structure).
- `pools`: The set of caches used for searching.

Note 1: Callbacks are not executed inside parallel blocks
Note 2: Callbacks will be ignored if `callbacks=nothing`

"""
function Base.append!(
        index::SearchGraph,
        db;
        parallel_block=1,
        parallel_minimum_first_block=parallel_block,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )
    db = convert(AbstractDatabase, db)
    append!(index.db, db)

    parallel_block == 1 && return _sequential_append_loop!(index, callbacks, pools)

    n = length(index) + length(db)
    m = 0

    parallel_minimum_first_block = min(parallel_minimum_first_block, n)
    while length(index) < parallel_minimum_first_block
        m += 1
        push_item!(index, db[m], false, callbacks, pools)
    end

    sp = length(index) + 1
    sp > n && return index

    resize!(index.links, n)
    _parallel_append_loop!(index, pools, sp, n, parallel_block, callbacks)
end

"""
    index!(index::SearchGraph; parallel_block=1, parallel_minimum_first_block=parallel_block, callbacks=SearchGraphCallbacks())

Indexes the already initialized database (e.g., given in the constructor method). It can be made in parallel or sequentially.
The arguments are the same than `append!` function but using the internal `index.db` as input.

"""
function index!(
        index::SearchGraph;
        parallel_block=1,
        parallel_minimum_first_block=parallel_block,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )
    @assert length(index) == 0 && length(index.db) > 0
    parallel_block == 1 && return _sequential_append_loop!(index, callbacks, pools)

    m = 0
    db = index.db
    n = length(db)

    parallel_minimum_first_block = min(parallel_minimum_first_block, n)
    while length(index) < parallel_minimum_first_block
        m += 1
        push_item!(index, db[m], false, callbacks, pools)
    end

    sp = length(index) + 1
    sp > n && return index
    resize!(index.links, n)
    _parallel_append_loop!(index, pools, sp, n, parallel_block, callbacks)
end

function _sequential_append_loop!(index::SearchGraph, callbacks, pools::SearchGraphPools)
    i = length(index)
    n = length(index.db)
    while i < n
        i += 1
        push_item!(index, index.db[i], false, callbacks, pools)
    end

    index
end

function _connect_links(index, sp, ep)
    Threads.@threads for i in sp:ep
        @inbounds for id in index.links[i]
            lock(index.locks[id])
            try
                push!(index.links[id], i)
                # sat_should_push(index.links[id], index, index[i], i, -1.0) && push!(index.links[id], i)
            finally
                unlock(index.locks[id])
            end
        end
    end
end

function _parallel_append_loop!(index::SearchGraph, pools::SearchGraphPools, sp, n, parallel_block, callbacks)
    while sp < n
        ep = min(n, sp + parallel_block)
        index.verbose && rand() < 0.01 && println(stderr, "appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())

        # searching neighbors
        # @show length(index.links), length(index.db), length(db), length(index.locks), length(index), sp, ep
        Threads.@threads for i in sp:ep
            @inbounds index.links[i] = find_neighborhood(index, index.db[i], pools)
        end

        # connecting neighbors
        _connect_links(index, sp, ep)
        
        # increasing locks => new items are enabled for searching (and reported by length so they can also be hints)
        resize!(index.locks, ep)
        for i in sp:ep
            @inbounds index.locks[i] = Threads.SpinLock()
        end
        
        # apply callbacks
        callbacks !== nothing && execute_callbacks(callbacks, index, sp, ep)
        sp = ep + 1
    end

    index
end

"""
    push!(
        index::SearchGraph,
        item;
        push_item=true,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )

Appends an object into the index. It accepts the same arguments that `push!` but assuming some default values.

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `push_item`: if `push_item=false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)
- `callbacks`: The set of callbacks that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- `pools`: The set of caches used for searching.

- Note: `callbacks=nothing` ignores the execution of any callback
"""
function push!(
        index::SearchGraph,
        item;
        push_item=true,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )
    push_item!(index, item, push_item, callbacks, pools)
end

"""
push_item!(
    index::SearchGraph,
    item,
    push_item,
    callbacks,
    pools
)

Appends an object into the index

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `push_item`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)
- `callbacks`: The set of callbacks that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- `pools`: The set of caches used for searching.

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
function push_item!(
    index::SearchGraph,
    item,
    push_item,
    callbacks,
    pools
)
    neighbors = find_neighborhood(index, item, pools)
    push_neighborhood!(index, item, neighbors, callbacks; push_item)

    neighbors
end

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