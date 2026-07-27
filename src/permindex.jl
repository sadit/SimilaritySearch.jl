# This file is a part of SimilaritySearch.jl

export PermutedSearchIndex

"""
    PermutedSearchIndex(; index, π, π′=invperm(π))

Wraps a search `index` together with a permutation `π` of its identifiers, and defines the
related accessor functions. Applying a permutation to the underlying storage (e.g., so that
frequently co-accessed objects are stored close together) can improve cache efficiency; this
wrapper lets that reordering be applied without changing the identifiers seen by the
application.

# Keyword Arguments
- `index`: the wrapped search index.
- `π`: permutation mapping internal identifiers (as stored in `index`) to external identifiers.
- `π′`: inverse permutation, mapping external identifiers to internal identifiers in `index`; defaults to `invperm(π)`.

# Examples

```julia
π = shuffle(1:length(index))
p = PermutedSearchIndex(; index, π)
```
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

"""
    search(p::PermutedSearchIndex, ctx::AbstractContext, q, res) -> res

Solves query `q` against the wrapped `p.index`, then remaps each result's identifier from
internal (`p.index`) space to external (`p.π`) space, so callers always see identifiers
relative to the original, unpermuted dataset.
"""
function search(p::PermutedSearchIndex, ctx::AbstractContext, q, res)
    out = search(p.index, ctx, q, res)
    @inbounds for i in eachindex(res.items)
        x = res.items[i]
        res.items[i] = IdDist(p.π[x.id], x.dist)
    end

    out
end
