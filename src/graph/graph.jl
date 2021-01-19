# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using JSON
export LocalSearchAlgorithm, NeighborhoodAlgorithm, SearchGraph, find_neighborhood, push_neighborhood!, search_context, VisitedVertices, parallel_fit

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end

### Basic operations on the index
const OPTIMIZE_LOGBASE = 10
const OPTIMIZE_LOGBASE_STARTING = 4

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Dict{Int32, UInt8} #IdDict{Int32,UInt8}
function VisitedVertices(n)
    vstate = VisitedVertices()
    sizehint!(vstate, n)
    vstate
end

@inline getstate(vstate::VisitedVertices, i) = get(vstate, i, UNKNOWN)
@inline function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end

mutable struct SearchGraphOptions
    automatic_optimization::Bool
    recall::Float64
    ksearch::Int
    tol::Float64
    verbose::Bool
end

struct SearchGraph{DistType<:PreMetric, DataType<:AbstractVector, SearchType<:LocalSearchAlgorithm, NeighborhoodType<:NeighborhoodAlgorithm} <: AbstractSearchContext
    dist::DistType
    db::DataType
    links::Vector{Vector{Int32}}
    search_algo::SearchType
    neighborhood_algo::NeighborhoodType
    res::KnnResult
    opts::SearchGraphOptions
end

Base.copy(g::SearchGraph; dist=g.dist, db=g.db, links=g.links, search_algo=g.search_algo, neighborhood_algo=g.neighborhood_algo, res=g.res, opts=g.opts) =
    SearchGraph(dist, db, links, search_algo, neighborhood_algo, res, opts)

function SearchGraph(dist::PreMetric, db::AbstractVector;
        search_algo::LocalSearchAlgorithm=BeamSearch(),
        neighborhood_algo::NeighborhoodAlgorithm=LogNeighborhood(),
        automatic_optimization=false,
        recall=0.9,
        ksearch=10,
        tol=0.001,
        verbose=true)
    links = Vector{Int32}[]
    opts = SearchGraphOptions(automatic_optimization, recall, ksearch, tol, verbose)
    index = SearchGraph(dist, eltype(db)[], links, search_algo, neighborhood_algo, KnnResult(ksearch), opts)

    for item in db
        push!(index, item)
    end

    index
end

## function parallel_fit(::Type{SearchGraph}, dist::PreMetric, dataset::AbstractVector{T}; firstblock=100_000, block=10_000, recall=0.9, k=10, search_algo=BeamSearch(), neighborhood_algo=LogSatNeighborhood(1.1), automatic_optimization=false, verbose=true) where T
##     links = Vector{Int32}[]
##     index = SearchGraph(T[], recall, k, links, search_algo, neighborhood_algo, verbose)
##     firstblock = min(length(dataset), firstblock)
## 
##     for i in 1:firstblock
##         push!(index, dist, dataset[i]; automatic_optimization=automatic_optimization)
##     end
## 
##     sp = length(index.db)
##     n = length(dataset)
##     N = Vector(undef, block)
##     while sp < n
##         ep = min(n, sp + block)
##         m = ep - sp
##         CTX = [search_context(index) for i in 1:Threads.nthreads()]
##         KNN = [KnnResult(1) for i in 1:Threads.nthreads()]
##         Threads.@threads for i in 1:m
##             searchctx = CTX[Threads.threadid()]
##             knn = KNN[Threads.threadid()]
##             reset!(searchctx, n=length(index.db))
##             N[i] = find_neighborhood(index, dist, dataset[sp+i], knn, searchctx)
##         end
##         
##         @show (sp, ep, n, length(N), index.search_algo, index.neighborhood_algo, Dates.now())
##         for i in 1:m
##             sp += 1
##             push_neighborhood!(index, dataset[sp], N[i])
##         end
##     end
## 
##     index
## end


include("opt.jl")

## neighborhoods
include("neighborhood/fixedneighborhood.jl")
include("neighborhood/logneighborhood.jl")
include("neighborhood/logsatneighborhood.jl")
include("neighborhood/satneighborhood.jl")
include("neighborhood/vorneighborhood.jl")

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
If hints is given then these vertices will be used as starting poiints for the search process.
"""
function search(index::SearchGraph, q, res::KnnResult)
    length(index.db) > 0 && search(index.search_algo, index, q, res)
    res
end

"""
    optimize!(perf::Performance, index::SearchGraph;
    recall=0.9, ksearch=10, verbose=index.opts.verbose, tol::Float64=0.01, maxiters::Integer=3, probes::Integer=0) 
    optimize!(perf, index.search_algo, index; recall=recall, tol=tol, maxiters=3, probes=probes)

Optimizes the index for the specified kind of queries.
"""
function optimize!(perf::Performance, index::SearchGraph;
    recall=index.opts.recall, ksearch=index.opts.ksearch, verbose=index.opts.verbose, tol::Float64=index.opts.tol, maxiters::Integer=3, probes::Integer=0) 
    optimize!(perf, index.search_algo, index; recall=recall, tol=tol, maxiters=3, probes=probes)
end
