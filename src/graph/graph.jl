# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using JSON
export LocalSearchAlgorithm, NeighborhoodAlgorithm, SearchGraph, find_neighborhood, push_neighborhood!, search_context, VisitedVertices

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end

### Basic operations on the index
const OPTIMIZE_LOGBASE = 10
const OPTIMIZE_LOGBASE_STARTING = 4

mutable struct SearchGraph{T} <: Index
    db::Vector{T}
    recall::Float64
    k::Int
    links::Vector{Vector{Int32}}
    search_algo::LocalSearchAlgorithm
    neighborhood_algo::NeighborhoodAlgorithm
    verbose::Bool
end

function search_context(index::SearchGraph)
    search_context(index.search_algo, length(index.db))
end

#=
@enum VisitedVertexState begin
    UNKNOWN = 0
    VISITED = 1
    EXPLORED = 2
end =#

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Dict{Int32, UInt8} #IdDict{Int32,UInt8}
function VisitedVertices(n::Int)
    vstate = VisitedVertices()
    sizehint!(vstate, ceil(Int, sqrt(n)))
    vstate
end

@inline getstate(vstate::VisitedVertices, i) = get(vstate, i, UNKNOWN)
@inline function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end

#=
const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Vector{UInt8}
VisitedVertices(n::Int) = zeros(UInt8, n)
getstate(vstate::VisitedVertices, i) = vstate[i]
function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end
=#

function fit(::Type{SearchGraph}, dist, dataset::AbstractVector{T}; recall=0.9, k=10, search_algo=BeamSearch(), searchctx=nothing, neighborhood_algo=LogSatNeighborhood(1.1), automatic_optimization=true, verbose=true) where T
    links = Vector{Int32}[]
    index = SearchGraph(T[], recall, k, links, search_algo, neighborhood_algo, verbose)
    knn = KnnResult(1)
    searchctx = searchctx === nothing ? search_context(index) : searchctx

    for item in dataset
        reset!(searchctx, n=length(index.db))
        push!(index, dist, item, knn; automatic_optimization=automatic_optimization, searchctx=searchctx)
    end

    index
end

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
# include("tihc.jl")
include("beamsearch.jl")


"""
    find_neighborhood(index::SearchGraph{T}, dist, item)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be
its neighbors (intenal function)
"""
function find_neighborhood(index::SearchGraph, dist, item, knn::KnnResult, searchctx)
    n = length(index.db)
    neighbors = Int32[]
    n > 0 && neighborhood(index.neighborhood_algo, index, dist, item, knn, neighbors, searchctx)
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
    push!(index::SearchGraph, dist, item)

Inserts `item` into the index.
"""
function push!(index::SearchGraph, dist, item, knn::KnnResult=KnnResult(1); automatic_optimization=true, searchctx=nothing)
    searchctx = searchctx === nothing ? search_context(index) : searchctx
    neighbors = find_neighborhood(index, dist, item, knn, searchctx)
    push_neighborhood!(index, item, neighbors)
    n = length(index.db)

    if automatic_optimization && n > OPTIMIZE_LOGBASE_STARTING
        k = ceil(Int, log(OPTIMIZE_LOGBASE, 1+n))
        k1 = ceil(Int, log(OPTIMIZE_LOGBASE, 2+n))
        k != k1 && optimize!(index, dist, recall=index.recall)
    end

    if index.verbose && length(index.db) % 5000 == 0
        println(stderr, "added n=$(length(index.db)), neighborhood=$(length(neighbors)), $(now())")
    end
    knn, neighbors
end

const EMPTY_INT_VECTOR = Int[]

"""
    search(index::SearchGraph, dist, q, res::KnnResult; hints)  

Solves the specified query `res` for the query object `q`.
If hints is given then these vertices will be used as starting poiints for the search process.
"""
function search(index::SearchGraph, dist, q, res::KnnResult; searchctx=nothing)
    searchctx = searchctx === nothing ? search_context(index) : searchctx
    length(index.db) > 0 && search(index.search_algo, index, dist, q, res, searchctx)
    res
end

"""
    optimize!(index::SearchGraph, dist;
        recall=0.9,
        k=10,
        num_queries=128,
        perf=nothing,
        tol::Float64=0.01,
        maxiters::Int=3,
        probes::Int=0)

Optimizes the index for the specified kind of queries.
"""
function optimize!(index::SearchGraph{T}, dist;
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
