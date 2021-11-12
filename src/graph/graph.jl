# This file is a part of SimilaritySearch.jl

using Dates
export LocalSearchAlgorithm, SearchGraph, SearchGraphOptions, VisitedVertices, NeighborhoodReduction, SatNeighborhood, IdentityNeighborhood, find_neighborhood, push_neighborhood!
export Callback, RandomHintsCallback, DisjointNeighborhoodHints

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
- The \$in\$ size is computed as \$\\Delta in\$, i.e., this is not searched in the current index yet for accepting future edges.
- reduce is intended to postprocess neighbors (after search process, i.e., once out edges are computed); do not change \$k\$ but always must return a copy of the reduced result set.

Note: The underlying graph is undirected, in and out edges are fused in the same priority queue; old edges can be discarded when closer elements are found.
Note: Set \$logbase=Inf\$ to obtain a fixed number of \$in\$ nodes; and set \$minsize=0\$ to obtain a pure logarithmic growing neighborhood.

"""
@with_kw mutable struct Neighborhood
    k::Int32 = 2 # actual neighborhood
    ksearch::Int32 = 2
    logbase::Float32 = 2
    minsize::Int32 = 2
    Δ::Float32 = 1
    reduce::NeighborhoodReduction = IdentityNeighborhood()
end

Base.copy(N::Neighborhood; k=N.k, ksearch=N.ksearch, logbase=N.logbase, minsize=N.minsize, Δ=N.Δ, reduce=copy(N.reduce)) =
    Neighborhood(; k, ksearch, logbase, minsize, Δ, reduce)

struct NeighborhoodCallback <: Callback end

"""
    struct SearchGraph <: AbstractSearchContext

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `search_algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

- `hints`: Initial points for exploration (empty hints imply using random points)

Note: Parallel insertions should be made through `append!` function with `parallel_block > 1`

"""
@with_kw struct SearchGraph{DistType<:PreMetric, DataType<:AbstractVector, SType<:LocalSearchAlgorithm}<:AbstractSearchContext
    dist::DistType = SqL2Distance()
    db::DataType = Vector{Float32}[]
    links::Vector{KnnResult{Int32,Float32}} = KnnResult{Int32,Float32}[]
    locks::Vector{Threads.SpinLock} = Threads.SpinLock[]
    hints::Vector{Int32} = Int32[]
    search_algo::SType = BeamSearch()
    neighborhood::Neighborhood = Neighborhood()
    callbacks::Dict{Symbol,Callback} = Dict(
        #:parameters => OptimizeParametersCallback(),
        :hints => RandomHintsCallback(),
        :neighborhood => NeighborhoodCallback()
    )
    callback_logbase::Float32 = 1.5
    callback_starting::Int32 = 8
    verbose::Bool = true
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
        callbacks=g.callbacks,
        callback_logbase=g.callback_logbase,
        callback_starting=g.callback_starting,
        verbose=true
) = SearchGraph(; dist, db, links, locks, hints, search_algo, neighborhood, callbacks, callback_logbase, callback_starting, verbose)

## search algorithms

include("ihc.jl")
include("beamsearch.jl")
## parameter optimization and neighborhood definitions
include("opt.jl")
include("neighborhood.jl")
include("hints.jl")

"""
    append!(index::SearchGraph, db; parallel_block=1, parallel_minimum_first_block=parallel_block, apply_callbacks=true)

Appends all items in db to the index. It can be made in parallel or sequentially.
In case of a parallel appending, then:
- `parallel_block` must be bigger than 1 and describes the batch size to append in parallel (i.e., in the order of thousands,
  depending on the size of the `db` and the number of available threads).
- `parallel_minimum_first_block` indicates the minimum number of items inserted sequentially before going parallel, it can be 0 if the index is already
  populated, defaults to `parallel_block`.

Note: Parallel doesn't trigger callbacks inside blocks.
"""
function Base.append!(index::SearchGraph, db;
    parallel_block=1, parallel_minimum_first_block=parallel_block, apply_callbacks=true)

    if parallel_block == 1
        for item in db
            push!(index, item)
        end

        return index
    end

    m = 0
    n = length(index) + length(db)
    while length(index) < parallel_minimum_first_block
        m += 1
        push!(index, db[m])
    end

    sp = length(index) + 1
    sp > n && return index

    resize!(index.db, n)
    resize!(index.links, n)
    for i in sp:n
        m += 1
        index.db[i] = db[m]
    end

    while sp < n
        ep = min(n, sp + parallel_block)
        index.verbose && println(stderr, "appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())

        # searching neighbors
        # @show length(index.links), length(index.db), length(db), length(index.locks), length(index), sp, ep
        Threads.@threads for i in sp:ep
           @inbounds index.links[i] = find_neighborhood(index, index.db[i], getknnresult(), getvisitedvertices(index))
        end

        # connecting neighbors
        k = index.neighborhood.k
        Threads.@threads for i in sp:ep
            @inbounds for (id, dist) in index.links[i]
                lock(index.locks[id])
                try
                    vertex = index.links[id]
                    vertex.k = max(maxlength(vertex), k)
                    push!(vertex, i => dist)
                finally
                    unlock(index.locks[id])
                end
            end
        end

        # increasing locks => new items are enabled for searching (and reported by length so they can also be hints)
        resize!(index.locks, ep)
        for i in sp:ep
            @inbounds index.locks[i] = Threads.SpinLock()
        end
        
        # apply callbacks
        apply_callbacks && callbacks(index, sp, ep)
        sp = ep + 1
    end

    index
end

"""
    push!(index::SearchGraph, item)

Appends `item` into the index.
"""
function push!(index::SearchGraph, item)
    neighbors = find_neighborhood(index, item, getknnresult(), getvisitedvertices(index))
    push_neighborhood!(index, item, neighbors)
    neighbors
end


"""
    search(index::SearchGraph, q, res::KnnResult; hints=index.hints, vstate=nothing)

Solves the specified query `res` for the query object `q`.
"""
function search(index::SearchGraph, q, res::KnnResult; hints=index.hints, vstate=nothing)
    vstate = vstate === nothing ? getvisitedvertices(index) : vstate

    if length(index) > 0
        search(index.search_algo, index, q, res, hints, vstate)
    end
    
    res
end

"""
    callback(opt::NeighborhoodCallback, index)

SearchGraph's callback for adjusting neighborhood strategy
"""
function callback(opt::NeighborhoodCallback, index)
    N = index.neighborhood
    N.ksearch = ceil(Int, N.minsize + log(N.logbase, length(index)))
    N.k = N.ksearch + ceil(Int, N.ksearch * N.Δ)
end

"""
    callbacks(index::SearchGraph, n=length(index), m=n+1)

Process all registered callbacks in `index`
"""
function callbacks(index::SearchGraph, n=length(index), m=n+1)
    if n >= index.callback_starting && ceil(Int, log(index.callback_logbase, n)) != ceil(Int, log(index.callback_logbase, m))
        for (name, callback_object) in index.callbacks
            index.verbose && println(stderr, "calling callback ", name, "; n=$(length(index)), type=", typeof(callback_object))
            callback(callback_object, index)
        end
    end
end

