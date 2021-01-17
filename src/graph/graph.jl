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

struct SearchGraph{DistType<:PreMetric, DataType<:AbstractVector, SearchType<:LocalSearchAlgorithm, NeighborhoodType<:NeighborhoodAlgorithm} <: AbstractSearchContext
    dist::DistType
    db::DataType
    links::Vector{Vector{Int32}}
    search_algo::SearchType
    neighborhood_algo::NeighborhoodType
    res::KnnResult
    recall::Float64
    k::Int
    verbose::Bool
end

function SearchGraph(dist::PreMetric, db::AbstractVector;
        search_algo::LocalSearchAlgorithm=BeamSearch(),
        neighborhood_algo::NeighborhoodAlgorithm=SatNeighborhood(1.1),
        automatic_optimization=false,
        recall=0.9,
        k=10,
        verbose=true)
    links = Vector{Int32}[]
    index = SearchGraph(dist, eltype(db)[], links, search_algo, neighborhood_algo, recall, KnnResult(k), k, verbose)

    for item in db
        push!(index, item; automatic_optimization=automatic_optimization)
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
include("neighborhood/gallopingneighborhood.jl")
include("neighborhood/satneighborhood.jl")
include("neighborhood/galsatneighborhood.jl")
include("neighborhood/vorneighborhood.jl")

## search algorithms
include("ihc.jl")
include("beamsearch.jl")


"""
    find_neighborhood(index::SearchGraph{T}, item)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be
its neighbors (intenal function)
"""
function find_neighborhood(index::SearchGraph, item)
    n = length(index.db)
    n > 0 && neighborhood(index.neighborhood_algo, index, item)
    neighbors
end

"""
    push_neighborhood!(index::SearchGraph, item, L::AbstractVector{Int32})

Inserts the object `item` into the index, i.e., creates an edge from items listed in L and the
vertex created for ìtem` (internal function)
"""
function push_neighborhood!(index::SearchGraph{T}, item::T, L::Vector{Int32}) where T
    push!(index.db, item)
    n = length(index.db)
    for objID in L
        push!(index.links[objID], n)
    end

    push!(index.links, L)
end

"""
    push!(index::SearchGraph, item; automatic_optimization=index.automatic_optimization)

Appends `item` into the index.
"""
function push!(index::SearchGraph, item; automatic_optimization=index.automatic_optimization)
    neighbors = find_neighborhood(index, item)
    push_neighborhood!(index, item, neighbors)
    n = length(index.db)

    if automatic_optimization && n > OPTIMIZE_LOGBASE_STARTING
        k = ceil(Int, log(OPTIMIZE_LOGBASE, 1+n))
        k1 = ceil(Int, log(OPTIMIZE_LOGBASE, 2+n))
        k != k1 && optimize!(index, recall=index.recall)
    end

    if index.verbose && length(index.db) % 10000 == 0
        println(stderr, "added n=$(length(index.db)), neighborhood=$(length(neighbors)), $(index.search_algo), $(index.neighborhood_algo), $(now())")
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
    optimize!(index::SearchGraph, dist::PreMetric;
        recall=0.9,
        k=10,
        num_queries=128,
        perf=nothing,
        tol::Float64=0.01,
        maxiters::Int=3,
        probes::Int=0)

Optimizes the index for the specified kind of queries.
"""
function optimize!(index::SearchGraph{T};
    recall=0.9,
    k=10,
    num_queries=128,
    perf=nothing,
    tol::Float64=0.01,
    maxiters::Int=3,
    probes::Int=0) where T
    if perf === nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end
    optimize!(index.search_algo, index, dist, recall, perf, tol=tol, maxiters=3, probes=probes)
end
