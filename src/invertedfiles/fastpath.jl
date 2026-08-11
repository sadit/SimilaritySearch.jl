# This file is part of InvertedFiles.jl

export set_distance_evaluate

"""
    set_distance_evaluate(dist::PreMetric, intersection::Integer, size1::Integer, size2::Integer)

Computes a score for a candidate found while merging posting lists, given the intersection size of the
matching posting lists and the total number of non-zero entries of each of the two compared elements
(`size1`, `size2`). For a handful of distances (see [`has_exact_fastpath`](@ref)) this is the *exact*
distance value, computed purely from these three integers, with no need to touch the original objects.
For any other `dist` it returns a cheap monotonic proxy (the `Dist.Sets.Intersection` formula), used only
to rank candidates during the merge; `search` follows up with a [`rerank!`](@ref) pass against the raw
stored objects to compute the true `dist` in that case.
"""
set_distance_evaluate(::Dist.Sets.Intersection, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0f0 - intersection / max(size1, size2)
set_distance_evaluate(::Dist.Sets.Dice, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0f0 - (2intersection) / (size1 + size2)
set_distance_evaluate(::Dist.Sets.Jaccard, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0f0 - (intersection) / (size1 + size2 - intersection)
set_distance_evaluate(::Dist.Sets.CosineSet, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0f0 - (intersection) / (sqrt(Float32(size1)) * sqrt(Float32(size2)))

function set_distance_evaluate(rt::Dist.Sets.RogersTanimoto, intersection::Int32, size1::Int32, size2::Int32)::Float32
    tt = intersection
    tf = size1 - tt
    ft = size2 - tt
    ff = rt.σ - tt - tf - ft
    1.0f0 - Float32(tt + ff) / Float32(tt + ff + 2 * (tf + ft))
end

# generic fallback: a cheap monotonic proxy for any distance without a closed-form case above;
# `search` gates a `rerank!` pass on `has_exact_fastpath` to recover the true distance.
set_distance_evaluate(::PreMetric, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0f0 - intersection / max(size1, size2)

set_distance_evaluate(t, intersection::Integer, size1::Integer, size2::Integer)::Float32 =
    set_distance_evaluate(t, convert(Int32, intersection), convert(Int32, size1), convert(Int32, size2))

"""
    has_exact_fastpath(dist::PreMetric)::Bool

Whether the score computed while merging posting lists (via [`set_distance_evaluate`](@ref) for
set-adjacency indexes, or the inlined dot product for weighted-adjacency indexes) is already the exact
`dist` value. When `false`, `search` follows up with a [`rerank!`](@ref) pass against the objects stored
in the index's `db` to compute the true distance over the candidates found during the merge.
"""
has_exact_fastpath(::Dist.Sets.Intersection) = true
has_exact_fastpath(::Dist.Sets.Dice) = true
has_exact_fastpath(::Dist.Sets.Jaccard) = true
has_exact_fastpath(::Dist.Sets.CosineSet) = true
has_exact_fastpath(::Dist.Sets.RogersTanimoto) = true
has_exact_fastpath(::Dist.NormCosine) = true
has_exact_fastpath(::PreMetric) = false
