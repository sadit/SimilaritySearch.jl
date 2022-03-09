# This file is a part of SimilaritySearch.jl
export HausdorffDistance, MinHausdorffDistance

"""
    HausdorffDistance(dist::SemiMetric)

Hausdorff distance is defined as the maximum of the minimum between two clouds of points.

```math 
Hausdorff(U, V) = \\max{\\max_{u \\in U} nndist(u, V), \\max{v \\in V} nndist(v, U) }
```

where ``nndist(u, V)`` computes the distance of ``u`` to its nearest neighbor in ``V`` using the `dist` metric.
"""
struct HausdorffDistance{Dtype<:SemiMetric} <: SemiMetric
    dist::Dtype
end

function _exhaustive_nndist(dist::SemiMetric, u::T, V) where T
    min_ = typemax(eltype(u))

    @inbounds for j in eachindex(V)
        min_ = min(evaluate(dist, u, V[j]), min_)
    end

    min_
end

function _hausdorff1(dist::SemiMetric, u, v)
    s = 0.0
    @inbounds for i in eachindex(u)
        s = max(s, _exhaustive_nndist(dist, u[i], v))
    end

    s
end

"""
    evaluate(m::HausdorffDistance, u, v)

Computes the Hausdorff distance between two cloud of points.

`u` and `v` are iterables where each object can be measured with the internal distance `dist`
"""
function evaluate(m::HausdorffDistance, u, v)
    if  length(u) == 1 || length(v) == 1
        _hausdorff1(m.dist, u, v)
    else
        max(_hausdorff1(m.dist, u, v), _hausdorff1(m.dist, v, u))
    end
end

"""
    MinHausdorffDistance(dist::SemiMetric)

Similar to [HausdorffDistance](@ref) but using minimum instead of maximum.
"""
struct MinHausdorffDistance{Dtype<:SemiMetric} <: SemiMetric
    dist::Dtype
end

function _minhausdorff1(dist::SemiMetric, u, v)
    s = 0.0
    @inbounds for i in eachindex(u)
        s = min(s, _exhaustive_nndist(dist, u[i], v))
    end

    s
end

"""
    evaluate(m::MinHausdorffDistance, u, v)

Computes a variant of the Hausdorff distance that uses the minimum instead of the maximum. `u` and `v` are iterables where each object can be measured with the internal distance `dist`
"""
function evaluate(m::MinHausdorffDistance, u, v)
    if  length(u) == 1 || length(v) == 1
        _minhausdorff1(m.dist, u, v)
    else
        min(_minhausdorff1(m.dist, u, v), _minhausdorff1(m.dist, v, u))
    end
end