# This file is a part of SimilaritySearch.jl
export SearchGraphContext

"""
function SearchGraphContext(KnnType::Type{<:AbstractKnn}=KnnSorted;
        logger=LogList(AbstractLog[InformativeLog(1.0)]),
        verbose=false,
        neighborhood=Neighborhood(SatNeighborhood()),
        hints_callback=KCentersHints(; logbase=1.2),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=4Threads.nthreads(),
        logbase_callback=1.5,
        starting_callback=256,
        beams = zeros(IdDist, 96, 3 * Threads.maxthreadid()),
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
- `beams`: Knn queues cache for insertions
- `vstates`: visited vertices cache 
- `verbose`: controls the number of output messages

#Notes
- The callbacks are triggers that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- The search graph is composed of direct and reverse links, direct links are controled with a `neighborhood`
    object, mostly use to control how neighborhoods are refined. Reverse links are created when a vertex appears in the neighborhood of another vertex
- `parallel_block`: The number of elements that the multithreading algorithm process at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `A set of caches to alleviate memory allocations in `SearchGraph` construction and searching. Relevant on multithreading scenarious where distance functions, `evaluate`
can call other metric indexes that can use these shared resources (globally defined).

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
verbose(ctx::SearchGraphContext) = ctx.verbose
knnqueue(::SearchGraphContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)

@inline function getvstate(len::Integer, ctx::SearchGraphContext)
    reuse!(ctx.vstates[Threads.threadid()], len)
end

@inline function getbeam(nsize::Integer, ctx::SearchGraphContext)
    nsize = min(nsize, size(ctx.beams, 1))
    colID = Threads.threadid()
    knnqueue(KnnSorted, view(ctx.beams, 1:nsize, colID))
end

