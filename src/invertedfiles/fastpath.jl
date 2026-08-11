# This file is part of InvertedFiles.jl

export set_distance_evaluate

"""
    set_distance_evaluate(dist::PreMetric, intersection::Integer, size1::Integer, size2::Integer)

Computes a score for a candidate found while merging posting lists, given the intersection size of the
matching posting lists and the total number of non-zero entries of each of the two compared elements
(`size1`, `size2`). Only defined for the handful of distances with an exact closed form (see
[`has_exact_fastpath`](@ref)) — the resulting value is exact, computed purely from these three integers,
with no need to touch the original objects. For any other `dist`, `search_invfile` does not call this
function at all — it evaluates `dist` directly against the stored objects for every merge candidate
instead (see `FallbackInvFileOutput` in `invfilesearch.jl`); use `t > 1` to bound how many such
evaluations happen per query.
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

set_distance_evaluate(t, intersection::Integer, size1::Integer, size2::Integer)::Float32 =
    set_distance_evaluate(t, convert(Int32, intersection), convert(Int32, size1), convert(Int32, size2))

"""
    has_exact_fastpath(dist::PreMetric)::Bool

Whether the score computed while merging posting lists (via [`set_distance_evaluate`](@ref)) is
already the exact `dist` value. When `false`, `search_invfile` instead evaluates `dist` directly
against the objects stored in the index's `db` for every merge candidate — see
`FallbackInvFileOutput` in `invfilesearch.jl`; raise `t` above the default `1` to bound how many such
evaluations happen per query.
"""
has_exact_fastpath(::Dist.Sets.Intersection) = true
has_exact_fastpath(::Dist.Sets.Dice) = true
has_exact_fastpath(::Dist.Sets.Jaccard) = true
has_exact_fastpath(::Dist.Sets.CosineSet) = true
has_exact_fastpath(::Dist.Sets.RogersTanimoto) = true
has_exact_fastpath(::PreMetric) = false
