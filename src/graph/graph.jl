# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
using Dates
export LocalSearchAlgorithm, SearchGraph, SearchGraphOptions, VisitedVertices, NeighborhoodReduction, SatNeighborhood, IdentityNeighborhood, find_neighborhood, push_neighborhood!

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

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Dict{Int32, UInt8}

@inline getstate(vstate::VisitedVertices, i) = get(vstate, i, UNKNOWN)
@inline function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end

@with_kw mutable struct OptimizeParametersCallback <: Callback
    error = :distance # :recall, :distance, :distance_and_searchtime
    ksearch::Int32 = 10
    numqueries::Int32 = 32
    initialpopulation::Int32 = 4
    maxpopulation::Int32 = 4
    tol::Float32 = 0.01
    maxiters::Int32 = 4
end

@with_kw mutable struct RandomHintsCallback <: Callback
    logbase::Float32 = 1.5
end

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

Note 1: Parallel searches need copies of the structure or passing a KnnResult per thread.
Note 2: Parallel insertions should be made through `append!` function with `parallel=true`

"""
@with_kw struct SearchGraph{DistType<:PreMetric, DataType<:AbstractVector, SType<:LocalSearchAlgorithm}<:AbstractSearchContext
    dist::DistType = SqL2Distance()
    db::DataType = Vector{Float32}[]
    links::Vector{KnnResult{Int32,Float32}} = KnnResult{Int32,Float32}[]
    search_algo::SType = BeamSearch()
    neighborhood::Neighborhood = Neighborhood()
    res::KnnResult = KnnResult(10)

    callbacks::Dict{Symbol,Callback} = Dict(
        #:parameters => OptimizeParametersCallback(),
        :hints => RandomHintsCallback(),
        :neighborhood => NeighborhoodCallback()
    )
    callback_logbase::Float32 = 1.5
    callback_starting::Int32 = 8
    verbose::Bool = true
end

Base.copy(g::SearchGraph;
        dist=g.dist,
        db=g.db,
        links=g.links,
        search_algo=copy(g.search_algo),
        neighborhood=copy(g.neighborhood),
        res=KnnResult(maxlength(g.res)),
        callbacks=g.callbacks,
        callback_logbase=g.callback_logbase,
        callback_starting=g.callback_starting,
        verbose=true
    ) =
    SearchGraph(; dist, db, links, search_algo, neighborhood, res, callbacks, callback_logbase, callback_starting, verbose)


## search algorithms
include("ihc.jl")
include("beamsearch.jl")
## parameter optimization and neighborhood definitions
include("opt.jl")
include("neighborhood.jl")

"""
    append!(index::SearchGraph, db; parallel=false, parallel_firstblock=30_000, parallel_block=10_000, apply_callbacks=true)

Appends all items in db to the index. It can be made in parallel or sequentially.
In case of a parallel appending, then `parallel_firstblock` indicates the minimum
number of items before going parallel, and `parallel_block` sets the chunck size
to append in parallel.

Note: Parallel construction doesn't trigger callbacks listed in `callbacks', they must be executed manually.
"""
function Base.append!(index::SearchGraph, db;
        parallel=false, parallel_firstblock=30_000, parallel_block=10_000, apply_callbacks=true)

    if parallel
        parallel_firstblock = min(length(db), parallel_firstblock)
        for i in 1:parallel_firstblock
            push!(index, db[i])
        end

        sp = length(index) + 1
        n = length(db)

        INDEXES = [copy(index) for i in 1:Threads.nthreads()]
        
        while sp < n
            ep = min(n, sp + parallel_block)
            index.verbose && println(stderr, "appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now(), "; index=", index)
            X = @view db[sp:ep]
            parallel_append!(index, INDEXES, X)
            apply_callbacks && callbacks(index)
            sp = ep + 1
        end
    else
        for item in db
            push!(index, item)
        end
    end

    index

end

"""
    parallel_append!(index, INDEXES::Vector{<:SearchGraph}, X::AbstractVector)

Insert all items in `X` into the set of indexes. All indexes are _views_ of the same index;
callbacks are not called here. Internal function.
"""
function parallel_append!(index, INDEXES::Vector{<:SearchGraph}, X::AbstractVector)
    m = length(X)
    N = Vector{eltype(index.links)}(undef, m)
    Threads.@threads for i in 1:m
        I = INDEXES[Threads.threadid()]
        I.neighborhood.k = index.neighborhood.k
        I.neighborhood.ksearch = index.neighborhood.ksearch
        N[i] = find_neighborhood(I, X[i])
    end

    for i in 1:m
        push_neighborhood!(index, X[i], N[i]; apply_callbacks=false)
    end
end

"""
    callbacks(index::SearchGraph)

Process all registered callbacks in `index`
"""
function callbacks(index::SearchGraph)
    n = length(index)

    if n >= index.callback_starting
        k = ceil(Int, log(index.callback_logbase, 1+n))
        k1 = ceil(Int, log(index.callback_logbase, 2+n))
        if k != k1
            for (name, callback_object) in index.callbacks
                index.verbose && println(stderr, "calling callback ", name, "; n=$n")
                callback(callback_object, index)
            end
        end
    end
end

"""
    push!(index::SearchGraph, item)

Appends `item` into the index.
"""
function push!(index::SearchGraph, item)
    neighbors = find_neighborhood(index, item)
    push_neighborhood!(index, item, neighbors)
    neighbors
end

"""
    search(index::SearchGraph, q, res::KnnResult; hints=index.search_algo.hints)

Solves the specified query `res` for the query object `q`.
"""
function search(index::SearchGraph, q, res::KnnResult; hints=index.search_algo.hints)
    length(index) > 0 && search(index.search_algo, index, q, res, hints)
    res
end

"""
    callback(opt::RandomHintsCallback, index)

SearchGraph's callback for selecting hints at random
"""
function callback(opt::RandomHintsCallback, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    sample = unique(rand(1:n, m))
    empty!(index.search_algo.hints)
    append!(index.search_algo.hints, sample)
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
