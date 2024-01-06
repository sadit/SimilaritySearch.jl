# This file is a part of SimilaritySearch.jl

export LocalSearchAlgorithm, SearchGraph, SearchGraphPools, SearchGraphSetup, index!, push_item!
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

"""
    SearchGraphSetup(;
        logger=InformativeLog(),
        neighborhood=Neighborhood(),
        hints_callback=DisjointHints(),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=get_parallel_block(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=8
    )

    Convenient constructor for the following struct:

    struct SearchGraphSetup
        logger
        neighborhood::Neighborhood
        hints_callback::Union{Nothing,Callback}
        hyperparameters_callback::Union{Nothing,Callback}
        logbase_callback::Float32
        starting_callback::Int32
        parallel_block::Int32
        parallel_first_block::Int32
    end


# Arguments
- `logger`: how to handle and log events, mostly for insertions for now
- `neighborhood`: specify how neighborhoods are computed, see [`Neighborhood`](@ref) for more info.
- `hints_callback`: A callback to compute hints, please check hits.jl for more info.
- `hyperparameters_callback`: A callback to compute search hyperparameters, see [`OptimizeParameters`](@ref) for more info.
- `logbase_callback`: A log base to control when to run callbacks
- `starting_callback`: When to start to run callbacks, minimum length to do it
- `parallel_block`: the size of the block that is processed in parallel
- `parallel_first_block`: the size of the first block that is processed in parallel

#Notes
- The callbacks are triggers  that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- The search graph is composed of direct and reverse links, direct links are controled with a `neighborhood`
    object, mostly use to control how neighborhoods are refined. Reverse links are created when a vertex appears in the neighborhood of another vertex
- `parallel_block`: The number of elements that the multithreading algorithm process at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `parallel_first_block`: The number of sequential appends before running parallel.
- Parallel doesn't trigger callbacks inside blocks.

"""
struct SearchGraphSetup
    logger
    neighborhood::Neighborhood
    hints_callback::Union{Nothing,Callback}
    hyperparameters_callback::Union{Nothing,Callback}
    logbase_callback::Float32
    starting_callback::Int32
    parallel_block::Int32
    parallel_first_block::Int32
end

function SearchGraphSetup(;
        logger=InformativeLog(),
        neighborhood=Neighborhood(),
        hints_callback=DisjointHints(),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=get_parallel_block(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=8
    )
 
    SearchGraphSetup(logger, neighborhood, hints_callback, hyperparameters_callback,
                     logbase_callback, starting_callback, parallel_block, parallel_first_block)
end



abstract type LocalSearchAlgorithm end

include("graph.jl")
include("log.jl")
include("callbacks.jl")
include("rebuild.jl")
include("insertions.jl")
include("io.jl")
