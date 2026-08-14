# This file is part of InvertedFiles.jl

module InvertedFiles
using ..Intersections
import ..SimilaritySearch:
    search, index!, append_items!, push_item!, database, distance, knnqueue,
    add_block_evaluations!, add_distance_evaluations!
using ..SimilaritySearch
using ..SimilaritySearch: Dist, AbstractContext, getminbatch, @BATCHES, AbstractLog, InformativeLog, AbstractDatabase, KnnSorted, IdDist, AbstractSearchIndex, KnnHeap, Accessors
using ..Special.Sparse: SparseVecView
using Distances: PreMetric

using Base.Threads: SpinLock

export InvertedFileContext, getcontext, DictInvertedFile
include("sortedintset.jl")
include("plists.jl")

struct InvertedFileContext{A,B} <: AbstractContext
    logger::AbstractLog
    parallel_block::Int
    maxbatches::Int
    batchid::Int
    scheduler::Symbol
    costdists::Vector{Int}
    costblocks::Vector{Int}
    positions::A
    buffer::B
end

function InvertedFileContext(;
        logger = InformativeLog(dt=1.0),
        maxbatches::Integer = 8Threads.nthreads(),
        parallel_block = maxbatches,
        batchid::Integer = 1,
        scheduler::Symbol = get_batch_scheduler(),
        costdists::Vector{Int} = zeros(Int, maxbatches),
        costblocks::Vector{Int} = zeros(Int, maxbatches),
        positions = [Vector{UInt32}(undef, 32) for _ in 1:maxbatches],
        keytype::Type = Any,
        buffer = [Vector{PostingList{Vector{UInt32}, keytype}}(undef, 32) for _ in 1:maxbatches],
    )

    InvertedFileContext(logger, parallel_block, convert(Int, maxbatches), convert(Int, batchid), scheduler,
                         costdists, costblocks, positions, buffer)
end

Accessors.ConstructionBase.constructorof(::Type{<:InvertedFileContext{A,B}}) where {A,B} = (args...) -> InvertedFileContext{A,B}(args...)

knnqueue(::InvertedFileContext, args...) = knnqueue(KnnSorted, args...)

include("invfile.jl")
include("fastpath.jl")
include("invfilesearch.jl")

getcontext(invfile::AbstractInvertedFile; kwargs...) = InvertedFileContext(; kwargs...)
getcontext(invfile::InvertedFile{<:Any, <:AdjDict{K}}; kwargs...) where K = InvertedFileContext(; keytype=K, kwargs...)
getcontext(; kwargs...) = InvertedFileContext(; kwargs...)

end
