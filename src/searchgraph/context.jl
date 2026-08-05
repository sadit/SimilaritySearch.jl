# This file is a part of SimilaritySearch.jl
export SearchGraphContext

"""
    SearchGraphContext(KnnType::Type{<:AbstractKnn}=KnnSorted,
        vstates=nothing;
        logger=LogList(AbstractLog[InformativeLog(dt=2.0)]),
        verbose=false,
        neighborhood=Neighborhood(filter=SatNeighborhood()),
        hints_callback=RandomHints(; logbase=1.1),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=4Threads.nthreads(),
        logbase_callback=1.5,
        starting_callback=256,
        maxbatches=8Threads.nthreads(),
        batchid=1,
        beams=nothing
    ) -> SearchGraphContext

    SearchGraphContext(ctx::SearchGraphContext; kwargs...) -> SearchGraphContext

Context object that stores configuration, callbacks, and pre-allocated caches used while
building and searching a [`SearchGraph`](@ref). It must be passed along to functions like
`index!`, `search`, `searchbatch`, and `optimize_index!`.

The first method builds a new context from scratch, selecting the priority-queue
implementation `KnnType` (e.g., `KnnSorted` or `KnnHeap`) used internally, and a per-batch
`vstates` cache of visited-vertices buffers (one entry per batch, up to `maxbatches`). The
second method (a copy constructor) creates a modified copy of an existing context `ctx`,
overriding only the given keyword arguments while reusing the same `KnnType` and `vstates`.

# Arguments
- `KnnType`: type of priority queue used for the internal knn caches (`beams`), defaults to `KnnSorted`.
- `vstates`: per-batch cache of visited-vertices buffers, one entry per batch (`nothing` builds
  a fresh one sized by `maxbatches`).

# Keyword Arguments
- `logger`: how to handle and log events, mostly for insertions for now.
- `verbose`: controls the number of output messages.
- `neighborhood`: specifies how neighborhoods are computed, see [`Neighborhood`](@ref) for more info.
- `hints_callback`: a callback to compute hints, please check `hints.jl` for more info.
- `hyperparameters_callback`: a callback to compute search hyperparameters, see [`OptimizeParameters`](@ref) for more info.
- `logbase_callback`: a log base to control when to run callbacks.
- `starting_callback`: when to start to run callbacks, minimum index length to do it.
- `parallel_block`: the size of the block that is processed in parallel.
- `maxbatches`: hard cap on the batch count used by [`getminbatch`](@ref) for operations driven
  by this context, and the capacity (number of columns/entries) of `vstates`/`beams` when they
  are built automatically. Defaults to `8 * Threads.nthreads()`.
- `batchid`: the batch slot this context is tagged with (indexes into `vstates`/`beams`). Not
  meaningful on the root context (always `1`) -- per-batch copies tagging the running
  `@batchid()` are minted internally via `@set ctx.batchid = @batchid()`, once per batch, not
  passed here directly.
- `beams`: knn queues cache used while inserting elements (used by [`BeamSearch`](@ref);
  `nothing` builds a fresh one sized by `maxbatches`).

Each of these keyword arguments is stored verbatim in the field of the same name.

# Notes
- The callbacks are triggers that are called whenever the index grows enough. They keep hyperparameters and structure in shape.
- The search graph is composed of direct and reverse links; direct links are controlled with a `neighborhood`
    object, mostly used to control how neighborhoods are refined. Reverse links are created when a vertex appears in the neighborhood of another vertex.
- `parallel_block`: The number of elements that the multithreading algorithm processes at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `beams` and `vstates` are caches that alleviate memory allocations in `SearchGraph` construction and searching, indexed by `batchid` (race-free under every [`@BATCHES`](@ref)
  scheduler, unlike the `Threads.threadid()`-indexing used before). Relevant on multithreading scenarios where distance functions, `evaluate`,
can call other metric indexes that can use these shared resources (globally defined).

# Examples
```julia
using SimilaritySearch

ctx = SearchGraphContext()                          # default configuration
ctx = SearchGraphContext(; verbose=true)             # verbose logging
ctx2 = SearchGraphContext(ctx; parallel_block=64)    # copy overriding one keyword
ctx3 = SearchGraphContext(; maxbatches=4Threads.nthreads())  # smaller batch-cache cap
```
"""
struct SearchGraphContext{KnnType,VSType} <: AbstractContext
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
    maxbatches::Int32
    batchid::Int32
    costdist::Vector{Int}
    costblk::Vector{Int}
end

function SearchGraphContext(
    KnnType::Type{<:AbstractKnn}=KnnSorted,
    vstates=nothing; # 2^15 * 64 elements without resizing, one entry per batch (up to maxbatches)
    #vstates=[Set{UInt32}() for _ in 1:maxbatches];
    logger=LogList(AbstractLog[InformativeLog(dt=2.0)]),
    verbose=false,
    neighborhood=Neighborhood(filter=SatNeighborhood()),
    hints_callback=RandomHints(; logbase=1.1),
    hyperparameters_callback=OptimizeParameters(MinRecall(0.97)),  # use high MinRecall to achieve good graph structures
    parallel_block=4Threads.nthreads(),
    logbase_callback=1.5,
    starting_callback=256,
    maxbatches::Integer=8Threads.nthreads(),
    batchid::Integer=1,
    beams=nothing,
    costdist=nothing,
    costblk=nothing
)
    vstates === nothing && (vstates = [Vector{UInt64}(undef, 2^15) for _ in 1:maxbatches])
    beams === nothing && (beams = zeros(IdDist, 32, maxbatches))
    costdist === nothing && (costdist = zeros(Int, maxbatches))
    costblk === nothing && (costblk = zeros(Int, maxbatches))

    SearchGraphContext{KnnType,typeof(vstates)}(logger, verbose, neighborhood,
        hints_callback, hyperparameters_callback,
        convert(Float32, logbase_callback),
        convert(Int32, starting_callback),
        convert(Int32, parallel_block),
        beams, vstates,
        convert(Int32, maxbatches), convert(Int32, batchid),
        costdist, costblk)
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
    vstates=ctx.vstates,
    maxbatches=ctx.maxbatches,
    batchid=ctx.batchid,
    costdist=ctx.costdist,
    costblk=ctx.costblk
) where {KnnType,VSType}

    SearchGraphContext{KnnType,typeof(vstates)}(logger, verbose, neighborhood,
        hints_callback, hyperparameters_callback,
        logbase_callback, starting_callback,
        parallel_block,
        beams, vstates, maxbatches, batchid,
        costdist, costblk)
end

# SearchGraphContext has a phantom type parameter (KnnType, not derivable from any field),
# so ConstructionBase's default reconstruction (used by Accessors.@set) can't infer it --
# this override makes `@set ctx.batchid = ...`/`@set ctx.maxbatches = ...` work.
Accessors.ConstructionBase.constructorof(::Type{<:SearchGraphContext{K,V}}) where {K,V} =
    (args...) -> SearchGraphContext{K,V}(args...)

"""
    getminbatch(ctx::AbstractContext, n::Int, nt::Int=Threads.nthreads(); blocks_per_thread::Int=8)

[`getminbatch`](@ref) overload that derives its `maxbatches` cap from `ctx.maxbatches`,
so that any batch count computed for operations driven by `ctx` never exceeds the
capacity of its per-batch caches (`vstates`/`beams`, for [`SearchGraphContext`](@ref)).
This is the preferred way to compute `minbatch` for any [`@BATCHES`](@ref) loop that has
a context object available.
"""
getminbatch(ctx::AbstractContext, n::Int, nt::Int=Threads.nthreads(); blocks_per_thread::Int=8) =
    getminbatch(n, nt; blocks_per_thread, maxbatches=Int(ctx.maxbatches))

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

Retrieves `ctx`'s visited-vertices cache slot (indexed by `ctx.batchid`, race-free
regardless of `@BATCHES` scheduler -- callers running inside a batch should pass a
per-batch context tagged via `@set ctx.batchid = @batchid()`, see [`@batchid()`](@ref)) from
`ctx.vstates`, resetting/resizing it so that it can track visits over `len` elements.
"""
@inline function getvstate(len::Integer, ctx::SearchGraphContext)
    reuse!(ctx.vstates[ctx.batchid], len)
end

"""
    getbeam(nsize::Integer, ctx::SearchGraphContext) -> AbstractKnn

Retrieves `ctx`'s preallocated beam slot (indexed by `ctx.batchid`; a `KnnSorted` queue
backed by `ctx.beams`, truncated to at most `nsize` elements) used internally by
[`BeamSearch`](@ref).
"""
@inline function getbeam(nsize::Integer, ctx::SearchGraphContext)
    nsize = min(nsize, size(ctx.beams, 1))
    knnqueue(KnnSorted, view(ctx.beams, 1:nsize, ctx.batchid))
end

