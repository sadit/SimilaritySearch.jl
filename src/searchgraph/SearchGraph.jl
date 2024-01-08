# This file is a part of SimilaritySearch.jl

export LocalSearchAlgorithm, SearchGraph, SearchGraphContext
export index!, push_item!
export Neighborhood, IdentityNeighborhood, DistalSatNeighborhood, SatNeighborhood
export find_neighborhood
export BeamSearch, BeamSearchSpace, Callback
export KDisjointHints, DisjointHints, RandomHints
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
    abstract type NeighborhoodReduction end
    
Postprocessing of a neighborhood using some criteria. Called from `find_neighborhood`
"""
abstract type NeighborhoodReduction end

"""
    Neighborhood(; logbase=2, minsize=2, reduce=SatNeighborhood())
    
Determines the size of the neighborhood, \$k\$ is adjusted as a callback, and it is intended to affect previously inserted vertices.
The neighborhood is designed to consider two components \$k=in+out\$, i.e. _in_coming and _out_going edges for each vertex.
- The \$out\$ size is computed as \$minsize + \\log(logbase, n)\$ where \$n\$ is the current number of indexed elements; this is computed searching
for \$out\$  elements in the current index.
- The \$in\$ size is unbounded.
- reduce is intended to postprocess neighbors (after search process, i.e., once out edges are computed); do not change \$k\$ but always must return a copy of the reduced result set.

Note: Set \$logbase=Inf\$ to obtain a fixed number of \$in\$ nodes; and set \$minsize=0\$ to obtain a pure logarithmic growing neighborhood.

"""
@with_kw struct Neighborhood{Reduction<:NeighborhoodReduction}
    logbase::Float32 = 2
    minsize::Int32 = 2
    reduce::Reduction = SatNeighborhood()
end

########################### SearchGraphContext

include("visitedvertices.jl")
include("context.jl")

abstract type LocalSearchAlgorithm end

include("graph.jl")

getcontext(::SearchGraph) = DEFAULT_SEARCH_GRAPH_CONTEXT[]

include("log.jl")
include("callbacks.jl")
include("rebuild.jl")
include("insertions.jl")
include("io.jl")
