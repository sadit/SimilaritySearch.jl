# This file is part of InvertedFiles.jl

export BinaryInvertedFile, set_distance_evaluate
const DistancesForBinaryInvertedFile = Union{Dist.Sets.Intersection,Dist.Sets.Dice,Dist.Sets.Jaccard,Dist.Sets.CosineSet}
"""
    struct BinaryInvertedFile <: AbstractInvertedFile

Creates a binary weighted inverted index. An inverted index is an sparse matrix representation optimized for computing `k` nn elements (columns) under some distance.

# Properties:

- `dist`: Distance function to be applied, valid values are: `Dist.Sets.Intersection()`, `Dist.Sets.Dice()`, `Dist.Sets.Jaccard()`, and `Dist.Sets.CosineSet()
- `lists`: posting lists (non-zero values of the rows in the matrix)
- `sizes`: number of non-zero values per object (number of non-zero values per column)
- `locks`: Per row locks for multithreaded construction
"""
struct BinaryInvertedFile{
            DistType<:DistancesForBinaryInvertedFile,
            AdjType<:AbstractAdjList
        } <: AbstractInvertedFile
    dist::DistType
    adj::AdjType
    sizes::Vector{UInt32}
end

function Base.show(io::IO, invfile::BinaryInvertedFile; prefix="", indent="\t")
    println(io, prefix, "BinaryInvertedFile:")
    prefix = indent * prefix
    println(io, prefix, "dist: ", invfile.dist)
    println(io, prefix, "length: ", length(invfile))
    println(io, prefix, "adj: ", typeof(invfile.adj))
end

distance(idx::BinaryInvertedFile) = idx.dist

"""
    set_distance_evaluate(dist::SemiMetric, intersection::Integer, size1::Integer, size2::Integer)

Computes the distance function `dist` on a [`BinaryInvertedFile`](@ref).
"""
set_distance_evaluate(::Dist.Sets.Intersection, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - intersection / max(size1, size2)
set_distance_evaluate(::Dist.Sets.Dice, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - (2intersection) / (size1 + size2)
set_distance_evaluate(::Dist.Sets.Jaccard, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - (intersection) / (size1 + size2 - intersection)
set_distance_evaluate(::Dist.Sets.CosineSet, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - (intersection) / (sqrt(size1) * sqrt(size2))
set_distance_evaluate(t, intersection::Integer, size1::Integer, size2::Integer)::Float32 = set_distance_evaluate(t, convert(Int32, intersection), convert(Int32, size1), convert(Int32, size2))
"""
    BinaryInvertedFile(vocsize::Integer, dist=Dist.Sets.Jaccard())

Creates an `BinaryInvertedFile` with the given vocabulary size and for the given distance function `dist`:

# Arguments:
- `vocsize`: the vocabulary size of the index
- `dist`: the distance function to be used in searches
"""
function BinaryInvertedFile(vocsize::Integer, dist=Dist.Sets.Jaccard())
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    BinaryInvertedFile(dist, resize!(AdjList(UInt32), vocsize), UInt32[])
end

function internal_push!(idx::BinaryInvertedFile, ctx::InvertedFileContext, tokenID, objID, _)
    add!(idx.adj, tokenID, (objID,))
end
