# This file is a part of SimilaritySearch.jl

export LocalSearchAlgorithm, SearchGraph, SearchGraphContext
export index!, push_item!
export BeamSearch, BeamSearchSpace, Callback
export KDisjointHints, DisjointHints, RandomHints, EpsilonHints, KCentersHints, AdjacentStoredHints, matrixhints
export warmupbuild
#export RandomPruning, KeepNearestPruning, SatPruning, prune!

"""
    get_parallel_block()

Used by SearchGraph insertion functions to solve `find_neighborhood!` in blocks. Small blocks are better to ensure quality; faster constructions will be achieved if `parallel_block` is a multiply of `Threads.nthreads()`

"""
get_parallel_block() = Threads.nthreads() == 1 ? 1 : 8 * Threads.nthreads()

"""
    abstract type Callback end

Abstract type to trigger callbacks after some number of insertions.
SearchGraph stores the callbacks in `callbacks` (a dictionary that associates symbols and callback objects);
A SearchGraph object controls when callbacks are fired using `callback_logbase` and `callback_starting`

"""
abstract type Callback end

"""
    abstract type NeighborhoodFilter end
    
Postprocessing of a neighborhood using some criteria. Called from `find_neighborhood!`
"""
abstract type NeighborhoodFilter end

"""
    Neighborhood(; logbase=2, minsize=2, neardup=typemin(Float32), filter=SatNeighborhood())
    
Determines the size of the neighborhood; it is adjusted as a callback exponentially.
More detailed, the insertion algorithm searches for ``log_\\text{logbase}(N) + minsize)`` in the index where ``N`` is the size of the index/dataset,
then these neighbors are filtered with `filter`. The algorithms use `neardup` to discard proximal items to be part of a neighborhood.

## Parameters
- `logbase=2`: logarithmic base to determine the number of neighbors to retrieve
- `minsize=2`: minimum number of elements to retrieve
- `neardup=typemin(Float32)`: distance to identify an element as duplicate (neardups could be ignored from neighborhoods)
- `filter=SatNeighborhood()`: strategy to reduce the number of neighbors

Note: Set \$logbase=Inf\$ to obtain a fixed number of \$in\$ nodes; and set \$minsize=0\$ to obtain a pure logarithmic growing neighborhood.

"""
@kwdef struct Neighborhood{NFILTER<:NeighborhoodFilter}
    logbase::Float32 = 1.3f0
    minsize::Int32 = Int32(2)
    neardup::Float32 = typemin(Float32)
    filter::NFILTER = SatNeighborhood()
end

function Base.show(io::IO, n::Neighborhood)
    print(io, "Neighborhood(logbase=", n.logbase, ", minsize=", n.minsize, ", neardup=", n.neardup, ", filter=", n.filter, ")")
end

########################### SearchGraphContext

include("visitedvertices.jl")
include("context.jl")

"""
    abstract type LocalSearchAlgorithm end

Base type for the local search algorithms used to solve queries over a [`SearchGraph`](@ref).
Concrete subtypes (e.g., [`BeamSearch`](@ref)) implement the strategy used to traverse the
graph while looking for the near neighbors of a query object.
"""
abstract type LocalSearchAlgorithm end

"""
    BeamSearch(; bsize::Integer=4, Δ::Real=1.0, maxvisits::Integer=10^6) -> BeamSearch

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.

# Keyword Arguments
- `bsize`: The size of the beam.
- `Δ`: Soft margin for accepting elements into the beam.
- `maxvisits`: Maximum number of node visits allowed while searching, useful for early stopping without convergence.

# Examples
```julia
using SimilaritySearch

algo = BeamSearch(; bsize=8, Δ=1.0, maxvisits=10^6)
G = SearchGraph(Dist.SqL2(), VectorDatabase(); algo=Ref(algo))
```
"""
struct BeamSearch <: LocalSearchAlgorithm
    bsize::Int32  # size of the search beam
    Δ::Float32  # soft-margin for accepting an element into the beam
    maxvisits::Int64 # maximum visits by search, useful for early stopping without convergence, very high by default
end

BeamSearch(; bsize=4, Δ=1.0, maxvisits=10^6) = BeamSearch(Int32(bsize), Float32(Δ), Int64(maxvisits))

function Base.show(io::IO, bs::BeamSearch)
    print(io, "BeamSearch(bsize=", bs.bsize, ", Δ=", bs.Δ, ", maxvisits=", bs.maxvisits, ")")
end


### Basic operations on the index

"""
    SearchGraph(dist::PreMetric, db::AbstractDatabase; adj=AdjList(UInt32), hints=UInt32[],
                  algo=Ref(BeamSearch()), len=Ref(zero(Int64))) -> SearchGraph

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

# Keyword Arguments
- `dist`: The distance function (a `PreMetric`) used to compare stored objects, e.g., `Dist.SqL2()`.
- `db`: The database of indexed objects, see [`AbstractDatabase`](@ref) (e.g., `MatrixDatabase`, `VectorDatabase`).
- `adj`: The adjacency list storing the graph's direct links between objects.
- `hints`: Initial points for exploration (empty hints imply using random points).
- `algo`: The local search algorithm used to solve queries, stored as a `Ref{BeamSearch}` (see [`BeamSearch`](@ref)).
- `len`: The number of stored elements, as a `Ref{Int64}`; use `length(index)` instead of accessing it directly.

Note: Parallel insertions should be made through `append!` or `index!` function with `parallel_block > 1`

# Examples
```julia
using SimilaritySearch
const Dist = SimilaritySearch.Dist

X = rand(Float32, 8, 10^4)          # 10^4 vectors of dimension 8
db = MatrixDatabase(X)

G = SearchGraph(Dist.SqL2(), db)
ctx = SearchGraphContext()
index!(G, ctx)                       # builds the graph (inserts all items in db)

q = rand(Float32, 8)
res = knnqueue(ctx, 8)                # a knn result set for k=8
search(G, ctx, q, res)                # solves a single query

Q = MatrixDatabase(rand(Float32, 8, 10^2))
knns = searchbatch(G, ctx, Q, 8)      # solves many queries at once
```
"""
struct SearchGraph{DIST<:PreMetric,
    DB<:AbstractDatabase,
    ADJ<:AbstractAdjList,
    HINTS,
} <: AbstractSearchIndex
    dist::DIST
    db::DB
    adj::ADJ
    hints::HINTS
    algo::Ref{BeamSearch}
    len::Ref{Int64}
end

"""

    SearchGraph(dist::PreMetric, db::AbstractDatabase; adj=AdjList(UInt32), hints=UInt32[], algo=Ref(BeamSearch()), len=Ref(zero(Int64)))

Creates a SearchGraph index structure with the given distance and dataset.
This function only creates the skeleton struct and you need to call `index!` to index the given dataset or populate it with `append_items!`
"""
function SearchGraph(dist::PreMetric, db::AbstractDatabase; adj=AdjList(UInt32), hints=UInt32[], algo=Ref(BeamSearch()), len=Ref(zero(Int64)))
    SearchGraph(dist, db, adj, hints, algo, len)
end


function Base.show(io::IO, index::SearchGraph; prefix="", indent="  ")
    println(io, prefix, "SearchGraph:")
    prefix = prefix * indent
    println(io, prefix, "dist: ", index.dist)
    println(io, prefix, "length: ", index.len[])
    println(io, prefix, "algo: ", index.algo[])
    println(io, prefix, "adj: ", typeof(index.adj))
    println(io, prefix, "hints: ", typeof(index.hints), ", length: ", length(index.hints))
    show(io, index.db; prefix, indent)
end

@inline Base.length(g::SearchGraph)::Int64 = g.len[]

include("beamsearch.jl")

## parameter optimization and neighborhood definitions
include("optbs.jl")
include("neighborhood.jl")
include("hints.jl")

"""
    search(index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue) -> AbstractMetricQueue

Solves the specified query `res` for the query object `q` using the `SearchGraph` index `index`.
It dispatches the work to the local search algorithm stored in `index.algo` (e.g., [`BeamSearch`](@ref)),
using `ctx` to access preallocated caches (visited-vertices state, beams) and configuration.
The result object `res` is updated in-place and also returned.

# Examples
```julia
using SimilaritySearch

# G::SearchGraph and ctx::SearchGraphContext already built and indexed
q = rand(Float32, 8)
res = knnqueue(ctx, 8)     # k=8 nearest neighbors
search(G, ctx, q, res)
```
"""
function search(index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue)
    vstate = getvstate(length(index), ctx)
    search(index.algo[], index, ctx, q, res, index.hints, vstate)
end

include("callbacks.jl")
include("rebuild.jl")
include("staticindexing.jl")
include("insertions.jl")
