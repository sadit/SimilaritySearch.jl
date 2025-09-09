# This file is a part of SimilaritySearch.jl
export SearchGraphContext

"""
    SearchGraphContext(;
        logger=InformativeLog(),
        verbose::Bool=false,
        minbatch::Int = 0,
        neighborhood=Neighborhood(),
        hints_callback=DisjointHints(),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=get_parallel_block(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=8,
        iknns = zeros(IdWeight, 96, Threads.maxthreadid()),
        beam = zeros(IdWeight, 32, Threads.maxthreadid()),
        sat = zeros(IdWeight, 64, Threads.maxthreadid()),
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
- `iknns`: insertion result cache
- `beam`: beam cache
- `sat`: sat's neighborhood result cache
- `vstates`: visited vertices cache 
- `minbatch`: Minimum number of queries solved per each thread, see [`getminbatch`](@ref)
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
struct SearchGraphContext <: AbstractContext
    logger
    minbatch::Int
    verbose::Bool
    neighborhood::Neighborhood
    hints_callback::Union{Nothing,Callback}
    hyperparameters_callback::Union{Nothing,Callback}
    logbase_callback::Float32
    starting_callback::Int32
    parallel_block::Int32
    parallel_first_block::Int32
    iknns::Matrix{IdWeight}
    beam::Matrix{IdWeight}
    sat::Matrix{IdWeight}
    # vstates::Vector{VisitedVerticesBits}
    vstates::Vector{Vector{UInt64}}
end

function SearchGraphContext(;
        logger=InformativeLog(),
        minbatch = 0,
        verbose = false,
        neighborhood=Neighborhood(SatNeighborhood(; nndist=3f-3)),
        hints_callback=KCentersHints(; logbase=1.2),
        hyperparameters_callback=OptimizeParameters(),
        parallel_block=4Threads.nthreads(),
        parallel_first_block=parallel_block,
        logbase_callback=1.5,
        starting_callback=256,
        iknns = zeros(IdWeight, 96, Threads.maxthreadid()),
        beam = zeros(IdWeight, 32, Threads.maxthreadid()),
        sat = zeros(IdWeight, 64, Threads.maxthreadid()),
        vstates = [Vector{UInt64}(undef, 32) for _ in 1:Threads.maxthreadid()]

    )
 
    SearchGraphContext(logger, minbatch, verbose, neighborhood,
                       hints_callback, hyperparameters_callback,
                       convert(Float32, logbase_callback),
                       convert(Int32, starting_callback),
                       convert(Int32, parallel_block),
                       convert(Int32, parallel_first_block),
                       iknns, beam, sat, vstates)
end

function SearchGraphContext(ctx::SearchGraphContext;
        logger=ctx.logger,
        minbatch=ctx.minbatch,
        verbose=ctx.verbose,
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
        vstates = ctx.vstates
    )
 
    SearchGraphContext(logger, minbatch, verbose, neighborhood,
                       hints_callback, hyperparameters_callback,
                       logbase_callback, starting_callback,
                       parallel_block, parallel_first_block,
                       iknns, beam, sat, vstates)
end

getminbatch(ctx::SearchGraphContext, n::Int=0) = getminbatch(ctx.minbatch, n)
verbose(ctx::SearchGraphContext) = ctx.verbose


# we use a static scheduler so,i.e., we use Polyester
@inline function getvstate(len::Integer, context::SearchGraphContext)
    reuse!(context.vstates[Threads.threadid()], len)
end

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


#@inline function knnview(nsize::Integer, knns::AbstractMatrix{IdWeight}, i=Threads.threadid())
#    view(knns, 1:_knnsize(nsize, knns), i)
#end
