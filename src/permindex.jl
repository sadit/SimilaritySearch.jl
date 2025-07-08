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

@inline getcontext(p::PermutedSearchIndex) = getcontext(p.index)
@inline database(p::PermutedSearchIndex) = SubDatabase(database(p.index), p.π′)
@inline database(p::PermutedSearchIndex, i) = database(p.index, p.π′[i])
@inline distance(p::PermutedSearchIndex) = distance(p.index)
@inline Base.length(p::PermutedSearchIndex) = length(p.index)

function search(p::PermutedSearchIndex, ctx::AbstractContext, res)
    out = search(p.index, ctx, q, res)
    @inbounds for i in eachindex(res.items)
        x = res.items[i]
        res.items[i] = IdWeight(p.π[x.id], x.weight)
    end

    out
end
