# This file is a part of SimilaritySearch.jl
export SearchGraphContext

"""
function SearchGraphContext(KnnType::Type{<:AbstractKnn}=KnnSorted;
        logger=LogList(AbstractLog[InformativeLog(1.0)]),
        expnt=0,
        verbose=false,
        neighborhood=Neighborhood(SatNeighborhood()),
        hints_callback=KCentersHints(; logbase=1.2),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=4Threads.nthreads(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=256,
        knns = zeros(IdWeight, 96, 3 * Threads.maxthreadid()),
        vstates = [Vector{UInt64}(undef, 32) for _ in 1:Threads.maxthreadid()]
    )    

# Arguments
- `logger`: how to handle and log events, mostly for insertions for now
- `neighborhood`: specify how neighborhoods are computed, see [`Neighborhood`](@ref) for more info.
- `hints_callback`: A callback to compute hints, please check hits.jl for more info.
- `hyperparameters_callback`: A callback to compute search hyperparameters, see [`OptimizeParameters`](@ref) for more info.
- `logbase_callback`: A log base to control when to run callbacks
- `starting_callback`: When to start to run callbacks, minimum length to do it
- `parallel_block`: the size of the block that is processed in parallel
- `parallel_first_block`: the size of the first block that is processed in parallel
- `knns`: Knn queues cache for insertions
- `vstates`: visited vertices cache 
- `expnt`: Increases the number of batches to be processed by this number (the base is the number of threads)
- `verbose`: controls the number of output messages

#Notes
- The callbacks are triggers that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- The search graph is composed of direct and reverse links, direct links are controled with a `neighborhood`
    object, mostly use to control how neighborhoods are refined. Reverse links are created when a vertex appears in the neighborhood of another vertex
- `parallel_block`: The number of elements that the multithreading algorithm process at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `parallel_first_block`: The number of sequential appends before running parallel.
- Parallel doesn't trigger callbacks inside blocks.
- `A set of caches to alleviate memory allocations in `SearchGraph` construction and searching. Relevant on multithreading scenarious where distance functions, `evaluate`
can call other metric indexes that can use these shared resources (globally defined).

"""
struct SearchGraphContext{KnnType} <: AbstractContext
    logger::AbstractLog
    expnt::Int
    verbose::Bool
    neighborhood::Neighborhood
    hints_callback::Union{Nothing,Callback}
    hyperparameters_callback::Union{Nothing,Callback}
    logbase_callback::Float32
    starting_callback::Int32
    parallel_block::Int32
    parallel_first_block::Int32
    knns::Matrix{IdWeight}
    vstates::Vector{Vector{UInt64}}
end

function SearchGraphContext(KnnType::Type{<:AbstractKnn}=KnnSorted;
    logger=LogList(AbstractLog[InformativeLog(1.0)]),
    expnt=0,
    verbose=false,
    neighborhood=Neighborhood(filter=SatNeighborhood()),
    hints_callback=KCentersHints(; logbase=1.2),
    hyperparameters_callback=OptimizeParameters(),
    parallel_block=4Threads.nthreads(),
    parallel_first_block=parallel_block,
    logbase_callback=1.5,
    starting_callback=256,
    knns=zeros(IdWeight, 96, 3 * Threads.maxthreadid()),
    vstates=[Vector{UInt64}(undef, 32) for _ in 1:Threads.maxthreadid()]
)

    SearchGraphContext{KnnType}(logger, expnt, verbose, neighborhood,
        hints_callback, hyperparameters_callback,
        convert(Float32, logbase_callback),
        convert(Int32, starting_callback),
        convert(Int32, parallel_block),
        convert(Int32, parallel_first_block),
        knns, vstates)
end

function SearchGraphContext(ctx::SearchGraphContext{KnnType};
    logger=ctx.logger,
    expnt=ctx.expnt,
    verbose=ctx.verbose,
    neighborhood=ctx.neighborhood,
    hints_callback=ctx.hints_callback,
    hyperparameters_callback=ctx.hyperparameters_callback,
    parallel_block=ctx.parallel_block,
    parallel_first_block=ctx.parallel_first_block,
    logbase_callback=ctx.logbase_callback,
    starting_callback=ctx.starting_callback,
    knns=ctx.knns,
    vstates=ctx.vstates
) where {KnnType}

    SearchGraphContext{KnnType}(logger, expnt, verbose, neighborhood,
        hints_callback, hyperparameters_callback,
        logbase_callback, starting_callback,
        parallel_block, parallel_first_block,
        knns, vstates)
end

getminbatch(ctx::SearchGraphContext, n::Int) = getminbatch(n, Threads.nthreads(), ctx.expnt)
verbose(ctx::SearchGraphContext) = ctx.verbose
knnqueue(::SearchGraphContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)

@inline function getvstate(len::Integer, ctx::SearchGraphContext)
    reuse!(ctx.vstates[Threads.threadid()], len)
end

@inline function getknnbuffer(ctx::SearchGraphContext{KnnType}, pos, nsize::Integer) where {KnnType}
    nsize = min(nsize, size(ctx.knns, 1))
    colID = (Threads.threadid() - 1) * 3 + pos

    knnqueue(KnnType, view(ctx.knns, 1:nsize, colID))
end

@inline function getbeam(nsize::Integer, ctx::SearchGraphContext)
    pos = 1
    nsize = min(nsize, size(ctx.knns, 2))
    colID = (Threads.threadid() - 1) * 3 + pos
    knnqueue(KnnSorted, view(ctx.knns, 1:nsize, colID))
end

@inline getsatknnresult(nsize::Integer, ctx::SearchGraphContext) = getknnbuffer(ctx, 2, nsize)
@inline getiknnresult(nsize::Integer, ctx::SearchGraphContext) = getknnbuffer(ctx, 3, nsize)

#@inline function knnview(nsize::Integer, knns::AbstractMatrix{IdWeight}, i=Threads.threadid())
#    view(knns, 1:_knnsize(nsize, knns), i)
#end
