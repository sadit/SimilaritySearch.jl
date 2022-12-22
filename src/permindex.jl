# This file is a part of SimilaritySearch.jl

export PermutedSearchIndex

"""
    PermutedSearchIndex()

Wraps a permuted search index and define related functions. A permuted index can improve cache efficiency and this wrapper can be used to apply them without modifying applications or metadata.
"""
struct PermutedSearchIndex{PermType<:AbstractVector,IndexType<:AbstractSearchIndex} <: AbstractSearchIndex
    index::IndexType
    π::PermType
    π′::PermType
end

PermutedSearchIndex(; index, π, π′=invperm(π)) = PermutedSearchIndex(index, π, π′)

@inline getpools(p::PermutedSearchIndex) = getpools(p.index)
@inline database(p::PermutedSearchIndex) = SubDatabase(database(p), p.π′)
@inline database(p::PermutedSearchIndex, i) = database(p, π′[i])
@inline distance(p::PermutedSearchIndex) = p.dist
@inline Base.length(p::PermutedSearchIndex) = length(p.cov)

function search(p::PermutedSearchIndex, q, res::KnnResult; pools=getpools(index))
    out = search(p.index, q, res; pools)
    @inbounds for i in eachindex(res.id)
        res.id[i] = p.π[res.id[i]]
    end

    out
end