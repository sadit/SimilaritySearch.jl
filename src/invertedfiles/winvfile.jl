# This file is part of InvertedFiles.jl

export WeightedInvertedFile

"""
    struct WeightedInvertedFile <: AbstractInvertedFile

An inverted index is a sparse matrix representation of with floating point weights, it supports only positive non-zero values.
This index is optimized to efficiently solve `k` nearest neighbors (cosine distance, using previously normalized vectors).

# Parameters

- `lists`: posting lists (non-zero id-elements in rows)
- `weights`: non-zero weights (in rows)
- `sizes`: number of non-zero values in each element (non-zero values in columns)
"""
struct WeightedInvertedFile{AdjType<:AbstractAdjList} <: AbstractInvertedFile
    adj::AdjType
    sizes::Vector{UInt32}  ## number of non zero elements per vector
end

function Base.show(io::IO, invfile::WeightedInvertedFile; prefix="", indent="\t")
    println(io, prefix, "WeightedInvertedFile:")
    prefix = indent * prefix
    println(io, prefix, "length: ", length(invfile))
    println(io, prefix, "adj: ", typeof(invfile.adj))
end

distance(idx::WeightedInvertedFile) = Dist.NormCosine()

"""
    WeightedInvertedFile(vocsize::Integer)

Convenient function to create an empty `WeightedInvertedFile` with the given vocabulary size.
"""
function WeightedInvertedFile(vocsize::Integer)
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    WeightedInvertedFile(
        resize!(AdjList(IdWeight), vocsize),
        Vector{UInt32}(undef, 0)
    )
end

function internal_push!(idx::WeightedInvertedFile, ctx::InvertedFileContext, tokenID, objID, weight)
    add!(idx.adj, tokenID, (IdWeight(objID, weight),))
end
