# This file is a part of SimilaritySearch.jl

export LocalSearchAlgorithm, SearchGraph, SearchGraphContext
export index!, push_item!
export Neighborhood, IdentityNeighborhood, DistalSatNeighborhood, SatNeighborhood
export find_neighborhood
export BeamSearch, BeamSearchSpace, Callback
export KDisjointHints, DisjointHints, RandomHints, EpsilonHints, KCentersHints, AdjacentStoredHints, matrixhints
export RandomPruning, KeepNearestPruning, SatPruning, prune!

"""
    get_parallel_block()

Used by SearchGraph insertion functions to solve `find_neighborhood` in blocks. Small blocks are better to ensure quality; faster constructions will be achieved if `parallel_block` is a multiply of `Threads.nthreads()`

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
    
Postprocessing of a neighborhood using some criteria. Called from `find_neighborhood`
"""
abstract type NeighborhoodFilter end

"""
    Neighborhood(; logbase=2, minsize=2, filter=SatNeighborhood())
    
Determines the size of the neighborhood, \$k\$ is adjusted as a callback, and it is intended to affect previously inserted vertices.
The neighborhood is designed to consider two components \$k=in+out\$, i.e. _in_coming and _out_going edges for each vertex.
- The \$out\$ size is computed as \$minsize + \\log(logbase, n)\$ where \$n\$ is the current number of indexed elements; this is computed searching
for \$out\$  elements in the current index.
- The \$in\$ size is unbounded.
- filter is intended to postprocess neighbors (after search process, i.e., once out edges are computed); do not change \$k\$ but always must return a copy of the filterd result set.

Note: Set \$logbase=Inf\$ to obtain a fixed number of \$in\$ nodes; and set \$minsize=0\$ to obtain a pure logarithmic growing neighborhood.

"""
@with_kw struct Neighborhood{NFILTER<:NeighborhoodFilter}
    logbase::Float32 = 2
    minsize::Int32 = 2
    filter::NFILTER = SatNeighborhood()
end

########################### SearchGraphContext

include("visitedvertices.jl")
include("context.jl")

abstract type LocalSearchAlgorithm end

"""
    BeamSearch(bsize::Integer=16, Δ::Float32)

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.

- `bsize`: The size of the beam.
- `Δ`: Soft margin for accepting elements into the beam
- `maxvisits`: MAximum visits while searching, useful for early stopping without convergence
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
    struct SearchGraph <: AbstractSearchIndex

SearchGraph index. It stores a set of points that can be compared through a distance function `dist`.
The performance is determined by the search algorithm `algo` and the neighborhood policy.
It supports callbacks to adjust parameters as insertions are made.

- `hints`: Initial points for exploration (empty hints imply using random points)

Note: Parallel insertions should be made through `append!` or `index!` function with `parallel_block > 1`
"""
@with_kw struct SearchGraph{DIST<:SemiMetric,
                            DB<:AbstractDatabase,
                            ADJ<:AbstractAdjacencyList,
                            HINTS,
                           } <: AbstractSearchIndex
    dist::DIST = SqL2Distance()
    db::DB = VectorDatabase()
    adj::ADJ = AdjacencyLists.AdjacencyList(UInt32)
    hints::HINTS = UInt32[]
    algo::Ref{BeamSearch} = BeamSearch()
    len::Ref{Int64} = Ref(zero(Int64))
end

@inline Base.length(g::SearchGraph)::Int64 = g.len[]

"""
    enqueue_item!(index::SearchGraph, q, obj, res, objID, vstate)

Internal function that evaluates the distance between a database object `obj` with id `objID` and the query `q`.
It helps to evaluate, mark as visited, and enqueue in the result set.
"""
@inline function enqueue_item!(index::SearchGraph, q, obj, res, objID, vstate)
    check_visited_and_visit!(vstate, convert(UInt64, objID)) && return res
    d = evaluate(distance(index), q, obj)
    push_item!(res, objID, d)
    res.costevals += 1
    res
end

include("beamsearch.jl")

## parameter optimization and neighborhood definitions
include("optbs.jl")
include("neighborhood.jl")
include("hints.jl")

"""
    search(index::SearchGraph, context::SearchGraphContext, q, res

Solves the specified query `res` for the query object `q`.
"""
function search(index::SearchGraph, context::SearchGraphContext, q, res::AbstractKnn)
    search(index.algo[], index, context, q, res, index.hints)
end

getcontext(::SearchGraph) = SearchGraphContext()

include("log.jl")
include("callbacks.jl")
include("rebuild.jl")
include("insertions.jl")
include("io.jl")
