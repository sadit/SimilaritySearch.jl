# This file is a part of SimilaritySearch.jl
export HausdorffDistance, ChamferDissimilarity

"""
    HausdorffDistance(dist::PreMetric)

Hausdorff distance is defined as the maximum of the minimum between two clouds of points.

```math 
Hausdorff(U, V) = \\max{\\max_{u \\in U} nndist(u, V), \\max{v \\in V} nndist(v, U) }
```

where ``nndist(u, V)`` computes the distance of ``u`` to its nearest neighbor in ``V`` using the `dist` metric.
"""
struct HausdorffDistance{D<:PreMetric} <: PreMetric
    dist::D
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
    ChamferDissimilarity(distance)

Computes the Chamfer dissimilarity between two point clouds


```math 
ChamferDissimilarity(U, V) = \\frac{1}{|U|}\\sum_{u \\in U} nndist(u, V) + \\frac{1}{|V|}\\sum_{v \\in V} nndist(v, U)
```

where ``nndist(u, V)`` computes the distance of ``u`` to its nearest neighbor in ``V`` using the `dist` metric.


"""
struct ChamferDissimilarity{D<:PreMetric} <: PreMetric
    dist::D
end

function evaluate(D::ChamferDissimilarity, U, V)
    vsum, usum = 0.0, 0.0

    for v in V
        vsum += _exhaustive_nndist(D.dist, v, U)
    end
    
    for u in U
        usum += _exhaustive_nndist(D.dist, u, V)
    end

    Float32(usum / length(U) + vsum / length(U))
end