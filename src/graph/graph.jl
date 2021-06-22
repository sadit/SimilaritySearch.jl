# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LocalSearchAlgorithm, NeighborhoodAlgorithm, SearchGraph, SearchGraphOptions, find_neighborhood, push_neighborhood!, search_context, VisitedVertices, parallel_fit

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end

### Basic operations on the index
const OPTIMIZE_LOGBASE = 10
const OPTIMIZE_LOGBASE_STARTING = 4

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Dict{Int32, UInt8} #IdDict{Int32,UInt8}

@inline getstate(vstate::VisitedVertices, i) = get(vstate, i, UNKNOWN)
@inline function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end

"""
    SearchGraphOptions(automatic_optimization::Bool, recall::Float64, ksearch::Int, tol::Float64, verbose::Bool)

Defines a number of options for the SearchGraph
"""
mutable struct SearchGraphOptions
    automatic_optimization::Bool
    recall::Float64
    ksearch::Int
    tol::Float64
    verbose::Bool
end

struct SearchGraph{DistType<:PreMetric, DataType<:AbstractVector, SType<:LocalSearchAlgorithm, NType<:NeighborhoodAlgorithm}<:AbstractSearchContext
    dist::DistType
    db::DataType
    links::Vector{Vector{Int32}}
    search_algo::SType
    neighborhood_algo::NType
    res::KnnResult
    opts::SearchGraphOptions
end

Base.copy(g::SearchGraph;
        dist=g.dist,
        db=g.db,
        links=g.links,
        search_algo=copy(g.search_algo),
        neighborhood_algo=copy(g.neighborhood_algo),
        res=KnnResult(maxlength(g.res)),
        opts=g.opts
    ) =
    SearchGraph(dist, db, links, search_algo, neighborhood_algo, res, opts)

Base.string(p::SearchGraphOptions) = "{SearchGraphOptions: ksearch=$(p.recall), automatic_optimization=$(p.automatic_optimization), recall=$(p.recall)}"
Base.string(p::SearchGraph) = "{SearchGraph: dist=$(p.dist), n=$(length(p.db)), search_algo=$(string(p.search_algo)), neighborhood_algo=$(typeof(p.neighborhood_algo)), knn=$(maxlength(p.res))}"

"""
    SearchGraph(dist::PreMetric,
        db::AbstractVector;
        search_algo::LocalSearchAlgorithm=BeamSearch(),
        neighborhood_algo::NeighborhoodAlgorithm=LogNeighborhood(),
        automatic_optimization=false,
        recall=0.9,
        ksearch=10,
        tol=0.001,
        parallel=false,
        firstblock=100_000,
        block=10_000, 
        verbose=true)

Creates a SearchGraph object, i.e., an index to perform approximate search on `db`
using the given search and neighbohood strategies. If `automatic_optimization` is true,
then the structure tries to reach the given `recall` under the given `ksearch`. The construction will use
all available threads if `parallel=true`.
"""
function SearchGraph(dist::PreMetric,
        db::AbstractVector;
        search_algo::LocalSearchAlgorithm=BeamSearch(),
        neighborhood_algo::NeighborhoodAlgorithm=LogNeighborhood(),
        automatic_optimization=false,
        recall=0.9,
        ksearch=10,
        tol=0.001,
        parallel=false,
        firstblock=100_000,
        block=10_000, 
        verbose=true)

    opts = SearchGraphOptions(automatic_optimization, recall, ksearch, tol, verbose)
    index = SearchGraph(dist, eltype(db)[], Vector{Int32}[], search_algo, neighborhood_algo, KnnResult(ksearch), opts)
    verbose && println(stderr, string(index), parallel ? ", parallel=$parallel; firstblock=$firstblock, block=$block" : "")

    if parallel
        firstblock = min(length(db), firstblock)
        for i in 1:firstblock
            push!(index, db[i])
        end

        sp = length(index.db) + 1
        n = length(db)

        INDEXES = [copy(index) for i in 1:Threads.nthreads()]
        
        while sp < n
            ep = min(n, sp + block)
            verbose && println(stderr, string(index), (sp=sp, ep=ep, n=n), Dates.now())
            X = @view db[sp:ep]
            parallel_push!(INDEXES, X)
            sp = ep + 1
        end
    else
        for item in db
            push!(index, item)
        end
    end

    index
end

function parallel_push!(INDEXES::Vector{S}, X::AbstractVector) where {S<:SearchGraph}
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

    if index.opts.automatic_optimization && n > OPTIMIZE_LOGBASE_STARTING
        k = ceil(Int, log(OPTIMIZE_LOGBASE, 1+n))
        k1 = ceil(Int, log(OPTIMIZE_LOGBASE, 2+n))
        if k != k1
            seq = ExhaustiveSearch(index.dist, index.db; ksearch=index.opts.ksearch)
            queries = index.db[ unique(rand(1:n, 32)) ]
            perf = Performance(seq, queries, index.opts.ksearch; popnearest=true)
            optimize!(perf, index, recall=index.opts.recall)
        end
    end

    if index.opts.verbose && length(index.db) % 10000 == 0
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
              recall=index.opts.recall,
              ksearch=index.opts.ksearch,
              verbose=index.opts.verbose,
              tol::Float64=index.opts.tol,
              maxiters::Integer=3,
              probes::Integer=0)

Optimizes the index for the specified kind of queries.
"""

function optimize!(perf::Performance,
              index::SearchGraph;
              recall=index.opts.recall,
              ksearch=index.opts.ksearch,
              verbose=index.opts.verbose,
              tol::Float64=index.opts.tol,
              maxiters::Integer=3,
              probes::Integer=0)

    optimize!(perf, index.search_algo, index; recall=recall, tol=tol, maxiters=3, probes=probes)
end
