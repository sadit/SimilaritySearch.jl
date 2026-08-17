# This file is part of SimilaritySearch.jl

"""
    SatContext(KnnType::Type{<:AbstractKnnQueue}=KnnSorted; kwargs...) -> SatContext

Context object for the approximate/autotuned SAT search variants ([`BeamSearchSat`](@ref),
[`PruningSat`](@ref), [`BeamSearchMultiSat`](@ref), [`PrunParSat`](@ref),
[`BeamSearchParSat`](@ref)). Provides per-batch (`ctx.batchid`-indexed) scratch caches, so
every `@BATCHES`-driven search is race-free regardless of scheduler:

- `beam_ids`/`beam_dists`: backing storage for a `KnnSorted` beam queue, retrieved via `getbeam`.
- `queues`: a plain `Vector{UInt32}` used as a DFS stack, retrieved via `getqueue`.
- `vstates`: a visited-vertices bitset, retrieved via `getvstate`.

# Keyword Arguments
- `maxbeam`: capacity of the per-batch beam cache. Must be `>=` the largest `bsize`
  explored by whatever `optimization_space` is used with `optimize_index!` -- `getbeam`
  clamps rather than resizing/erroring on mismatch.
- `maxbatches`, `batchid`, `scheduler`, `costdists`, `costblocks`: same meaning as in
  [`GenericContext`](@ref)/[`SearchGraphContext`](@ref).
"""
struct SatContext{KnnType} <: AbstractContext
    verbose::Bool
    logger::AbstractLog
    beam_ids::Matrix{UInt32}
    beam_dists::Matrix{Float32}
    queues::Vector{Vector{UInt32}}
    vstates::Vector{Vector{UInt64}}
    maxbatches::Int32
    batchid::Int32
    scheduler::Symbol
    costdists::Vector{Int}
    costblocks::Vector{Int}
end

function SatContext(KnnType::Type{<:AbstractKnnQueue}=KnnSorted;
    verbose::Bool=false,
    logger=InformativeLog(),
    maxbeam::Integer=64,
    maxbatches::Integer=8Threads.nthreads(),
    batchid::Integer=1,
    scheduler::Symbol=get_batch_scheduler(),
    beam_ids=nothing,
    beam_dists=nothing,
    queues=nothing,
    vstates=nothing,
    costdists=nothing,
    costblocks=nothing
)
    beam_ids   === nothing && (beam_ids   = zeros(UInt32,  maxbeam, maxbatches))
    beam_dists === nothing && (beam_dists = zeros(Float32, maxbeam, maxbatches))
    queues     === nothing && (queues     = [UInt32[] for _ in 1:maxbatches])
    vstates    === nothing && (vstates    = [Vector{UInt64}(undef, 2^10) for _ in 1:maxbatches])
    costdists  === nothing && (costdists  = zeros(Int, maxbatches))
    costblocks === nothing && (costblocks = zeros(Int, maxbatches))

    SatContext{KnnType}(verbose, logger, beam_ids, beam_dists, queues, vstates,
        convert(Int32, maxbatches), convert(Int32, batchid), scheduler, costdists, costblocks)
end

function SatContext(ctx::SatContext{KnnType};
    verbose=ctx.verbose, logger=ctx.logger,
    beam_ids=ctx.beam_ids, beam_dists=ctx.beam_dists,
    queues=ctx.queues, vstates=ctx.vstates,
    maxbatches=ctx.maxbatches, batchid=ctx.batchid,
    scheduler=ctx.scheduler, costdists=ctx.costdists, costblocks=ctx.costblocks
) where {KnnType}
    SatContext{KnnType}(verbose, logger, beam_ids, beam_dists, queues, vstates,
        convert(Int32, maxbatches), convert(Int32, batchid), scheduler, costdists, costblocks)
end

# phantom KnnType param -- needed for Accessors.@set ctx.batchid = ... to work
Accessors.ConstructionBase.constructorof(::Type{<:SatContext{K}}) where {K} =
    (args...) -> SatContext{K}(args...)

verbose(ctx::SatContext) = ctx.verbose
knnqueue(::SatContext{KnnType}, args...) where {KnnType<:AbstractKnnQueue} = knnqueue(KnnType, args...)

"""
    getbeam(nsize::Integer, ctx::SatContext) -> AbstractKnnQueue

Retrieves `ctx`'s preallocated beam slot (indexed by `ctx.batchid`), truncated to at most
`nsize` elements.
"""
@inline function getbeam(nsize::Integer, ctx::SatContext)
    nsize = min(nsize, size(ctx.beam_ids, 1))
    knnqueue(KnnSorted, view(ctx.beam_ids, 1:nsize, ctx.batchid), view(ctx.beam_dists, 1:nsize, ctx.batchid))
end

"""
    getqueue(ctx::SatContext) -> Vector{UInt32}

Retrieves `ctx`'s preallocated DFS-stack slot (indexed by `ctx.batchid`), emptied before use.
"""
@inline function getqueue(ctx::SatContext)
    q = ctx.queues[ctx.batchid]
    empty!(q)
    q
end

"""
    getvstate(len::Integer, ctx::SatContext) -> Vector{UInt64}

Retrieves `ctx`'s visited-vertices cache slot (indexed by `ctx.batchid`), resetting/resizing
it so that it can track visits over `len` elements.
"""
@inline getvstate(len::Integer, ctx::SatContext) = reuse!(ctx.vstates[ctx.batchid], len)
