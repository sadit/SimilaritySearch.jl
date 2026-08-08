# This file is part of InvertedFiles.jl

module InvertedFiles
using ..Intersections
import ..SimilaritySearch:
    search, index!, append_items!, push_item!, database, distance, knnqueue,
    add_block_evaluations!, add_distance_evaluations!
using ..SimilaritySearch
using ..SimilaritySearch: Dist, AbstractContext, getminbatch, @BATCHES, AbstractLog, InformativeLog, AbstractDatabase, KnnSorted, IdDist, AbstractSearchIndex, KnnHeap, Accessors

using Base.Threads: SpinLock

export InvertedFileContext, getcontext
include("idweight.jl")
include("sortedintset.jl")
include("plists.jl")

struct InvertedFileContext{A,B,C,D} <: AbstractContext
    logger::AbstractLog
    parallel_block::Int
    maxbatches::Int32
    batchid::Int32
    costdist::Vector{Int}
    costblk::Vector{Int}
    positions::A
    cont_u32::B
    cont_iw::C
    cont_iiw::D
    knns::Matrix{IdWeight}
end

function InvertedFileContext(;
        logger = InformativeLog(dt=1.0),
        parallel_block = 256,
        maxbatches::Integer = 8Threads.nthreads(),
        batchid::Integer = 1,
        costdist::Vector{Int} = zeros(Int, maxbatches),
        costblk::Vector{Int} = zeros(Int, maxbatches),
        positions = [Vector{UInt32}(undef, 32) for _ in 1:maxbatches],
        cont_u32 = [Vector{PostingList{Vector{UInt32}}}(undef, 32) for _ in 1:maxbatches],
        cont_iw = [Vector{PostingList{Vector{IdWeight}}}(undef, 32) for _ in 1:maxbatches],
        cont_iiw = [Vector{PostingList{Vector{IdIntWeight}}}(undef, 32) for _ in 1:maxbatches],
        knns = zeros(IdWeight, 64, maxbatches)
    )

    InvertedFileContext(logger, parallel_block, convert(Int32, maxbatches), convert(Int32, batchid),
                         costdist, costblk, positions, cont_u32, cont_iw, cont_iiw, knns)
end

Accessors.ConstructionBase.constructorof(::Type{<:InvertedFileContext{A,B,C,D}}) where {A,B,C,D} = (args...) -> InvertedFileContext{A,B,C,D}(args...)

knnqueue(::InvertedFileContext, arg) = knnqueue(KnnSorted, arg)

include("invfile.jl")
include("winvfile.jl")
include("binvfile.jl")
include("invfilesearch.jl")
include("winvfilesearch.jl")
include("binvfilesearch.jl")

getcontext(invfile::AbstractInvertedFile; kwargs...) = InvertedFileContext(; kwargs...)
getcontext(; kwargs...) = InvertedFileContext(; kwargs...)

end
