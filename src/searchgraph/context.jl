# This file is a part of SimilaritySearch.jl
export SearchGraphContext

"""
    SearchGraphContext(;
        logger=InformativeLog(),
        neighborhood=Neighborhood(),
        hints_callback=DisjointHints(),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=get_parallel_block(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=8
        knn::Vector{KnnResult} = [KnnResult(16) for _ in 1:Threads.nthreads()]
        beam::Vector{KnnResult} = [KnnResult(16) for _ in 1:Threads.nthreads()]
        sat::Vector{KnnResult} = [KnnResult(16) for _ in 1:Threads.nthreads()]
        vstates::Vector = [VisitedVerticesBits(32) for _ in 1:Threads.nthreads()]
        minbatch::Int = 0
    )
    
    SearchGraphContext(ctx::SearchGraphContext;
        logger=ctx.logger,
        neighborhood=ctx.neighborhood,
        hints_callback=ctx.hints_callback,
        hyperparameters_callback=ctx.hyperparameters_callback,
        parallel_block=ctx.parallel_block,
        parallel_first_block=ctx.parallel_first_block,
        logbase_callback=ctx.logbase_callback,
        starting_callback=ctx.starting_callback,
        knn = ctx.knn,
        beam = ctx.beam,
        sat = ctx.sat,
        vstates = ctx.vstates,
        minbatch = 0
    )
 

    Convenient constructors for the following struct:

    struct SearchGraphContext <: AbstractContext
        logger
        neighborhood::Neighborhood
        hints_callback::Union{Nothing,Callback}
        hyperparameters_callback::Union{Nothing,Callback}
        logbase_callback::Float32
        starting_callback::Int32
        parallel_block::Int32
        parallel_first_block::Int32
        knn::Vector{KnnResult} 
        beam::Vector{KnnResult} 
        sat::Vector{KnnResult} 
        vstates::Vector 
        minbatch::Int
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
- `iknns`: insertion result cache
- `beam`: beam cache
- `sat`: sat's neighborhood result cache
- `vstates`: visited vertices cache 
- `minbatch`: Minimum number of queries solved per each thread, see [`getminbatch`](@ref)

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
struct SearchGraphContext{KNNSET} <: AbstractContext
    logger
    neighborhood::Neighborhood
    hints_callback::Union{Nothing,Callback}
    hyperparameters_callback::Union{Nothing,Callback}
    logbase_callback::Float32
    starting_callback::Int32
    parallel_block::Int32
    parallel_first_block::Int32
    iknns::KNNSET
    beam::KNNSET
    sat::KNNSET
    # vstates::Vector{VisitedVerticesBits}
    vstates::Vector{Vector{UInt64}}
    minbatch::Int
end

function SearchGraphContext(;
        logger=InformativeLog(),
        neighborhood=Neighborhood(SatNeighborhood(; hfactor=0f0, nndist=3f-3)),
        #hints_callback=DisjointHints(),
        hints_callback=KCentersHints(; logbase=1.2),
        #hints_callback=EpsilonHints(quantile=1/64),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=4Threads.nthreads(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=256,
        iknns = zeros(IdWeight, 96, Threads.maxthreadid()),
        beam = zeros(IdWeight, 32, Threads.maxthreadid()),
        sat = zeros(IdWeight, 64, Threads.maxthreadid()),
        # vstates = [VisitedVerticesBits(32) for _ in 1:Threads.nthreads()],
        vstates = [Vector{UInt64}(undef, 32) for _ in 1:Threads.maxthreadid()],
        minbatch = 0
    )
 
    SearchGraphContext(logger, neighborhood,
                       hints_callback, hyperparameters_callback,
                       convert(Float32, logbase_callback),
                       convert(Int32, starting_callback),
                       convert(Int32, parallel_block),
                       convert(Int32, parallel_first_block),
                       iknns, beam, sat, vstates, minbatch)
end

function SearchGraphContext(ctx::SearchGraphContext;
        logger=ctx.logger,
        neighborhood=ctx.neighborhood,
        hints_callback=ctx.hints_callback,
        hyperparameters_callback=ctx.hyperparameters_callback,
        parallel_block=ctx.parallel_block,
        parallel_first_block=ctx.parallel_first_block,
        logbase_callback=ctx.logbase_callback,
        starting_callback=ctx.starting_callback,
        iknns =  ctx.iknns,
        beam = ctx.beam,
        sat = ctx.sat,
        vstates = ctx.vstates,
        minbatch = 0
    )
 
    SearchGraphContext(logger, neighborhood,
                       hints_callback, hyperparameters_callback,
                       logbase_callback, starting_callback,
                       parallel_block, parallel_first_block,
                       iknns, beam, sat, vstates, minbatch)
end

# we use a static scheduler so,i.e., we use Polyester
@inline function getvstate(len::Integer, context::SearchGraphContext)
    reuse!(context.vstates[Threads.threadid()], len)
end

_knnsize(nsize::Integer, cache) = min(nsize, size(cache, 1))

@inline function getbeam(nsize::Integer, context::SearchGraphContext)
    nsize = min(nsize, size(context.beam, 2))
    xknn(view(context.beam, 1:nsize, Threads.threadid()))
end

@inline function getsatknnresult(nsize::Integer, context::SearchGraphContext)
    nsize = min(nsize, size(context.sat, 2))
    xknn(view(context.sat, 1:nsize, Threads.threadid()))
end

@inline function getiknnresult(nsize::Integer, context::SearchGraphContext)
    nsize = min(nsize, size(context.iknns, 2))
    xknn(view(context.iknns, 1:nsize, Threads.threadid()))
end

knndefault(v) = knn(v)

#@inline function knnview(nsize::Integer, knns::AbstractMatrix{IdWeight}, i=Threads.threadid())
#    view(knns, 1:_knnsize(nsize, knns), i)
#end
