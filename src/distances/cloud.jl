# This file is a part of SimilaritySearch.jl
export HausdorffDistance, MinHausdorffDistance

raw"""
    HausdorffDistance(dist::PreMetric)

Hausdorff distance is defined as the maximum of the minimum between two clouds of points.

\[ Hausdorff(U, V) \max{ \max_{u \in U} nndist(u, V), \max{v \in V} nndist(v, U) } \]

where ``nndist(u, V)`` computes the distance of ``u`` to its nearest neighbor in ``V``.
"""
struct HausdorffDistance{Dtype<:PreMetric} <: PreMetric
    dist::Dtype
end

function _exhaustive_nndist(dist::PreMetric, u::T, V) where T
    min_ = typemax(eltype(u))

    @inbounds for j in eachindex(V)
        min_ = min(evaluate(dist, u, V[j]), min_)
    end

    min_
end

function _hausdorff1(dist::PreMetric, u, v)
    s = 0.0
    @inbounds for i in eachindex(u)
        s = max(s, _exhaustive_nndist(dist, u[i], v))
    end

    s
end

function SimilaritySearch.evaluate(m::HausdorffDistance, u, v)
    if  length(u) == 1 || length(v) == 1
        _hausdorff1(m.dist, u, v)
    else
        max(_hausdorff1(m.dist, u, v), _hausdorff1(m.dist, v, u))
    end
end

"""
    MinHausdorffDistance(dist::PreMetric)

Similar to Hausdorff distance but using minimum instead of maximum.
"""
struct MinHausdorffDistance{Dtype<:PreMetric} <: PreMetric
    dist::Dtype
end

function _minhausdorff1(dist::PreMetric, u, v)
    s = 0.0
    @inbounds for i in eachindex(u)
        s = min(s, _exhaustive_nndist(dist, u[i], v))
    end

    s
end

function SimilaritySearch.evaluate(m::MinHausdorffDistance, u, v)
    if  length(u) == 1 || length(v) == 1
        _minhausdorff1(m.dist, u, v)
    else
        min(_minhausdorff1(m.dist, u, v), _minhausdorff1(m.dist, v, u))
    end
end