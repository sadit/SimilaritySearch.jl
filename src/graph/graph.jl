# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LocalSearchAlgorithm, NeighborhoodAlgorithm, SearchGraph, SearchGraphOptions, find_neighborhood, push_neighborhood!, VisitedVertices

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end
abstract type Callback end

### Basic operations on the index

const OPTIMIZE_LOGBASE_STARTING = 4

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Dict{Int32, UInt8}

@inline getstate(vstate::VisitedVertices, i) = get(vstate, i, UNKNOWN)
@inline function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end

@with_kw struct OptimizeParametersCallback <: Callback
    recall::Float32 = 0.9
    tol::Float32 = 0.01
    ksearch::Int32 = 10
    numqueries::Int32 = 32
end

@with_kw struct RandomHintsCallback <: Callback
    logbase::Float32 = 2.0
end

@with_kw struct SearchGraph{DistType<:PreMetric, DataType<:AbstractVector, SType<:LocalSearchAlgorithm, NType<:NeighborhoodAlgorithm}<:AbstractSearchContext
    dist::DistType = SqL2Distance()
    db::DataType = Vector{Float32}[]
    links::Vector{Vector{Int32}} = Vector{Int32}[]
    search_algo::SType = BeamSearch()
    neighborhood_algo::NType = LogNeighborhood()
    res::KnnResult = KnnResult(10)
    
    callback_list::Dict{Symbol, Callback} = Dict(:optimize_parameters => OptimizeParametersCallback(), :optimize_hints => RandomHintsCallback())
    callback_logbase::Int32 = 2
    callback_starting::Int32 = 8
    verbose::Bool = true
end


Base.copy(g::SearchGraph;
        dist=g.dist,
        db=g.db,
        links=g.links,
        search_algo=copy(g.search_algo),
        neighborhood_algo=copy(g.neighborhood_algo),
        res=KnnResult(maxlength(g.res)),
        callback_list=g.callback_list,
        callback_logbase=g.callback_logbase,
        callback_starting=g.callback_starting,
        verbose=true
    ) =
    SearchGraph(; dist, db, links, search_algo, neighborhood_algo, res, callback_list, callback_logbase, callback_starting, verbose)


"""
    append!(index::SearchGraph, db; parallel=false, parallel_firstblock=30_000, parallel_block=10_000)

Appends all items in db to the index. It can be made in parallel or sequentially.
In case of a parallel appending, then `parallel_firstblock` indicates the minimum
number of items before going parallel, and `parallel_block` sets the chunck size
to append in parallel.

Note: Parallel construction doesn't trigger callbacks listed in `callback_list', they must be executed manually.
"""
function Base.append!(index::SearchGraph, db;
        parallel=false, parallel_firstblock=30_000, parallel_block=10_000)

    if parallel
        parallel_firstblock = min(length(db), parallel_firstblock)
        for i in 1:parallel_firstblock
            push!(index, db[i])
        end

        sp = length(index.db) + 1
        n = length(db)

        INDEXES = [copy(index) for i in 1:Threads.nthreads()]
        
        while sp < n
            ep = min(n, sp + parallel_block)
            index.verbose && println(stderr, string(index), (sp=sp, ep=ep, n=n), Dates.now())
            X = @view db[sp:ep]
            parallel_append!(INDEXES, X)
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
    parallel_append!(INDEXES::Vector{<:SearchGraph}, X::AbstractVector)

Insert all items in `X` into the set of indexes. All indexes are _views_ of the same index. Internal function.
"""
function parallel_append!(INDEXES::Vector{<:SearchGraph}, X::AbstractVector)
    m = length(X)
    N = Vector{Vector{Int32}}(undef, m)
    Threads.@threads for i in 1:m
        tid = Threads.threadid()
        N[i] = find_neighborhood(INDEXES[tid], X[i])
    end

    for i in 1:m
        push_neighborhood!(INDEXES[1], X[i], N[i])
    end
end


include("opt.jl")

## neighborhoods
include("neighborhood/fixedneighborhood.jl")
include("neighborhood/logneighborhood.jl")
include("neighborhood/logsatneighborhood.jl")
include("neighborhood/satneighborhood.jl")

## search algorithms
include("ihc.jl")
include("beamsearch.jl")


"""
    find_neighborhood(index::SearchGraph{T}, item)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be
its neighbors (intenal function)
"""
function find_neighborhood(index::SearchGraph, item)::Vector{Int32}
    n = length(index.db)
    n == 0 ? Int32[] : find_neighborhood(index.neighborhood_algo, index, item)
end

"""
    push_neighborhood!(index::SearchGraph, item, L::AbstractVector{Int32})

Inserts the object `item` into the index, i.e., creates an edge from items listed in L and the
vertex created for ìtem` (internal function)
"""
function push_neighborhood!(index::SearchGraph, item, L::Vector{Int32})
    push!(index.db, item)
    n = length(index.db)

    for objID in L
        push!(index.links[objID], n)
    end

    push!(index.links, L)
end

"""
    push!(index::SearchGraph, item)

Appends `item` into the index.
"""
function push!(index::SearchGraph, item)
    neighbors = find_neighborhood(index, item)
    push_neighborhood!(index, item, neighbors)
    n = length(index.db)

    if n >= index.callback_starting
        k = ceil(Int, log(index.callback_logbase, 1+n))
        k1 = ceil(Int, log(index.callback_logbase, 2+n))
        if k != k1
            for (name, callback_object) in index.callback_list
                index.verbose && println(stderr, "calling callback ", name, "; n=$n")
                callback(callback_object, index)
            end
        end
    end

    if index.verbose && length(index.db) % 10000 == 0
        println(stderr, "added n=$(length(index.db)), neighborhood=$(length(neighbors)), $(string(index.search_algo)), $(typeof(index.neighborhood_algo)), $(now())")
    end

    neighbors
end

"""
    search(index::SearchGraph, q, res::KnnResult)  

Solves the specified query `res` for the query object `q`.
"""
function search(index::SearchGraph, q, res::KnnResult)
    length(index.db) > 0 && search(index.search_algo, index, q, res)
    res
end

"""
    optimize!(perf::Performance,
              index::SearchGraph;
              recall=0.9,
              tol::Real=0.001,
              maxiters::Integer=3,
              probes::Integer=0)

Optimizes the index for the specified kind of queries.
"""

function optimize!(perf::Performance,
              index::SearchGraph;
              recall=0.9,
              tol::Real=0.001,
              maxiters::Integer=3,
              probes::Integer=0)

    optimize!(perf, index.search_algo, index; recall=recall, tol=tol, maxiters=maxiters, probes=probes)
end


"""
    callback(opt::RandomHintsCallback, index)

SearchGraph's callback for selecting hints at random
"""
function callback(opt::RandomHintsCallback, index)
    empty!(index.search_algo.hints)
    n = length(index.db)
    m = ceil(Int, log(opt.logbase, length(index.db)))
    sample = unique(rand(1:n, 2m))
    sample = sample[1:m]
    append!(index.search_algo.hints, sample)
end


"""
    callback(opt::OptimizeParametersCallback, index)

SearchGraph's callback for adjunting search parameters
"""
function callback(opt::OptimizeParametersCallback, index)
    seq = ExhaustiveSearch(index.dist, index.db; ksearch=opt.ksearch)
    sample = unique(rand(1:length(index.db), opt.numqueries))
    queries = index.db[sample]
    perf = Performance(seq, queries, opt.ksearch; popnearest=true)
    optimize!(perf, index, recall=opt.recall)
end
