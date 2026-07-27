# This file is a part of SimilaritySearch.jl
export SearchGraphContext

"""
    SearchGraphContext(KnnType::Type{<:AbstractKnn}=KnnSorted,
        vstates=[Vector{UInt64}(undef, 2^15) for _ in 1:Threads.maxthreadid()];
        logger=LogList(AbstractLog[InformativeLog(dt=2.0)]),
        verbose=false,
        neighborhood=Neighborhood(filter=SatNeighborhood()),
        hints_callback=RandomHints(; logbase=1.1),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=4Threads.nthreads(),
        logbase_callback=1.5,
        starting_callback=256,
        beams=zeros(IdDist, 32, Threads.maxthreadid())
    ) -> SearchGraphContext

    SearchGraphContext(ctx::SearchGraphContext; kwargs...) -> SearchGraphContext

Context object that stores configuration, callbacks, and pre-allocated caches used while
building and searching a [`SearchGraph`](@ref). It must be passed along to functions like
`index!`, `search`, `searchbatch`, and `optimize_index!`.

The first method builds a new context from scratch, selecting the priority-queue
implementation `KnnType` (e.g., `KnnSorted` or `KnnHeap`) used internally, and a per-thread
`vstates` cache of visited-vertices buffers. The second method (a copy constructor) creates a
modified copy of an existing context `ctx`, overriding only the given keyword arguments while
reusing the same `KnnType` and `vstates`.

# Arguments
- `KnnType`: type of priority queue used for the internal knn caches (`beams`), defaults to `KnnSorted`.
- `vstates`: per-thread cache of visited-vertices buffers, one entry per thread.

# Keyword Arguments
- `logger`: how to handle and log events, mostly for insertions for now.
- `verbose`: controls the number of output messages.
- `neighborhood`: specifies how neighborhoods are computed, see [`Neighborhood`](@ref) for more info.
- `hints_callback`: a callback to compute hints, please check `hints.jl` for more info.
- `hyperparameters_callback`: a callback to compute search hyperparameters, see [`OptimizeParameters`](@ref) for more info.
- `logbase_callback`: a log base to control when to run callbacks.
- `starting_callback`: when to start to run callbacks, minimum index length to do it.
- `parallel_block`: the size of the block that is processed in parallel.
- `beams`: knn queues cache used while inserting elements (used by [`BeamSearch`](@ref)).

Each of these keyword arguments is stored verbatim in the field of the same name.

# Notes
- The callbacks are triggers that are called whenever the index grows enough. They keep hyperparameters and structure in shape.
- The search graph is composed of direct and reverse links; direct links are controlled with a `neighborhood`
    object, mostly used to control how neighborhoods are refined. Reverse links are created when a vertex appears in the neighborhood of another vertex.
- `parallel_block`: The number of elements that the multithreading algorithm processes at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `beams` and `vstates` are caches that alleviate memory allocations in `SearchGraph` construction and searching. Relevant on multithreading scenarios where distance functions, `evaluate`,
can call other metric indexes that can use these shared resources (globally defined).

# Examples
```julia
using SimilaritySearch

ctx = SearchGraphContext()                          # default configuration
ctx = SearchGraphContext(; verbose=true)             # verbose logging
ctx2 = SearchGraphContext(ctx; parallel_block=64)    # copy overriding one keyword
```
"""
struct SearchGraphContext{KnnType, VSType} <: AbstractContext
    logger::AbstractLog
    verbose::Bool
    neighborhood::Neighborhood
    hints_callback::Union{Nothing,Callback}
    hyperparameters_callback::Union{Nothing,Callback}
    logbase_callback::Float32
    starting_callback::Int32
    parallel_block::Int32
    beams::Matrix{IdDist}
    vstates::VSType
    #vstates::Vector{Set{UInt32}}
end

function SearchGraphContext(
    KnnType::Type{<:AbstractKnn}=KnnSorted,
    vstates=[Vector{UInt64}(undef, 2^15) for _ in 1:Threads.maxthreadid()]; # 2^15 * 64 elements without resizing
    #vstates=[Set{UInt32}() for _ in 1:Threads.maxthreadid()];
    logger=LogList(AbstractLog[InformativeLog(dt=2.0)]),
    verbose=false,
    neighborhood=Neighborhood(filter=SatNeighborhood()),
    hints_callback=RandomHints(; logbase=1.1),
    hyperparameters_callback=OptimizeParameters(),
    parallel_block=4Threads.nthreads(),
    logbase_callback=1.5,
    starting_callback=256,
    beams=zeros(IdDist, 32, Threads.maxthreadid())    
)
    SearchGraphContext{KnnType,typeof(vstates)}(logger, verbose, neighborhood,
        hints_callback, hyperparameters_callback,
        convert(Float32, logbase_callback),
        convert(Int32, starting_callback),
        convert(Int32, parallel_block),
        beams, vstates)
end

function SearchGraphContext(ctx::SearchGraphContext{KnnType,VSType};
    logger=ctx.logger,
    verbose=ctx.verbose,
    neighborhood=ctx.neighborhood,
    hints_callback=ctx.hints_callback,
    hyperparameters_callback=ctx.hyperparameters_callback,
    parallel_block=ctx.parallel_block,
    logbase_callback=ctx.logbase_callback,
    starting_callback=ctx.starting_callback,
    beams=ctx.beams,
    vstates=ctx.vstates
) where {KnnType,VSType}

    SearchGraphContext{KnnType,typeof(vstates)}(logger, verbose, neighborhood,
        hints_callback, hyperparameters_callback,
        logbase_callback, starting_callback,
        parallel_block,
        beams, vstates)
end

#getminbatch(ctx::SearchGraphContext, n::Int) = getminbatch(n, Threads.nthreads())
"""
    verbose(ctx::SearchGraphContext) -> Bool

Returns whether `ctx` is configured to emit verbose output.
"""
verbose(ctx::SearchGraphContext) = ctx.verbose

"""
    knnqueue(ctx::SearchGraphContext{KnnType}, arg) -> AbstractKnn

Creates a knn priority queue of type `KnnType` (the type parameter stored in `ctx`), using `arg`
to initialize it (either an integer `k` or a preallocated vector), see [`knnqueue`](@ref).
"""
knnqueue(::SearchGraphContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)

"""
    getvstate(len::Integer, ctx::SearchGraphContext)

Retrieves the current thread's visited-vertices cache from `ctx.vstates`, resetting/resizing it
so that it can track visits over `len` elements.
"""
@inline function getvstate(len::Integer, ctx::SearchGraphContext)
    reuse!(ctx.vstates[Threads.threadid()], len)
end

"""
    getbeam(nsize::Integer, ctx::SearchGraphContext) -> AbstractKnn

Retrieves the current thread's preallocated beam (a `KnnSorted` queue backed by `ctx.beams`,
truncated to at most `nsize` elements) used internally by [`BeamSearch`](@ref).
"""
@inline function getbeam(nsize::Integer, ctx::SearchGraphContext)
    nsize = min(nsize, size(ctx.beams, 1))
    colID = Threads.threadid()
    knnqueue(KnnSorted, view(ctx.beams, 1:nsize, colID))
end

