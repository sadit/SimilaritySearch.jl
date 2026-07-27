# This file is a part of SimilaritySearch.jl
#export Hausdorff, Chamfer

"""
    Hausdorff(dist::PreMetric)

Hausdorff distance is defined as the maximum of the minimum between two clouds of points.

```math 
Hausdorff(U, V) = \\max{\\max_{u \\in U} nndist(u, V), \\max{v \\in V} nndist(v, U) }
```

where ``nndist(u, V)`` computes the distance of ``u`` to its nearest neighbor in ``V`` using the `dist` metric.
"""
struct Hausdorff{D<:PreMetric} <: PreMetric # is Hausdorff a metric when dist is not a metric?
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
    evaluate(m::Hausdorff, u, v)

Computes the Hausdorff distance between two cloud of points.

`u` and `v` are iterables where each object can be measured with the internal distance `dist`
"""
function evaluate(m::Hausdorff, u, v)
    if  length(u) == 1 || length(v) == 1
        _hausdorff1(m.dist, u, v)
    else
        max(_hausdorff1(m.dist, u, v), _hausdorff1(m.dist, v, u))
    end
end


"""
    Chamfer(distance)

Computes the Chamfer dissimilarity between two point clouds


```math 
Chamfer(U, V) = \\frac{1}{|U|}\\sum_{u \\in U} nndist(u, V) + \\frac{1}{|V|}\\sum_{v \\in V} nndist(v, U)
```

where ``nndist(u, V)`` computes the distance of ``u`` to its nearest neighbor in ``V`` using the `dist` metric.


"""
struct Chamfer{D<:PreMetric} <: PreMetric
    dist::D
end

function evaluate(D::Chamfer, U, V)
    vsum, usum = 0.0, 0.0

    for v in V
        vsum += _exhaustive_nndist(D.dist, v, U)
    end
    
    for u in U
        usum += _exhaustive_nndist(D.dist, u, V)
    end

    Float32(usum / length(U) + vsum / length(U))
end


"""
    EMD(dist, p)

Approximates the Earth Mover's Distance (EMD) between two point clouds `U` and `V` of
the same size as a greedy perfect matching: points are matched one at a time, each
being paired with its nearest still-unmatched point in the other cloud (measured with
`dist`), and the distances of the matched pairs are then combined with an ``L_p``-like
aggregation.

# Arguments
- `dist`: the base distance function used to compare individual points of the clouds.
- `p`: the exponent used to combine the distances of the matched pairs, i.e.,

```math
EMD(U, V) = \\left(\\sum_i evaluate(dist, u_i, v_{\\pi(i)})^p \\right)^{1/p}
```

where ``\\pi`` is the greedy matching found by the algorithm.

Note that this is a greedy approximation of the assignment problem underlying the exact
EMD (optimal transport), not the exact solution, and it requires `length(U) == length(V)`.

# Examples

```julia
d = Dist.Cloud.EMD(Dist.L2(), 2f0)
evaluate(d, U, V)
```
"""
struct EMD{D<:PreMetric} <: PreMetric  # is EMD a metric when dist is not a metric?
    dist::D
    p::Float32
end

"""
    evaluate(emd::EMD, U, V)

Computes the greedy perfect-matching approximation of the Earth Mover's Distance
between the point clouds `U` and `V` (see [`EMD`](@ref) for details).
"""
function evaluate(emd::EMD, U, V)
    n = length(U)
    s = 0f0
    C = collect(Int32, 1:n)  ## TODO cache this
    # t = rand() < 0.1
    # t &&  @info "===================="
    for i in eachindex(C)
        C[i] == 0 && break
        u = U[i]
        min_, argmin_ = typemax(Float32), 0
        for j in i:n
            objID = C[j]
            d = evaluate(emd.dist, u, V[objID])^emd.p

            if d < min_
                s += d
                min_, argmin_ = d, j
            end
        end

        C[argmin_], C[i] = C[i], C[argmin_]
        # t && @info (; pos="XXXX POST", C, i, min_, argmin_, s, n)
    end

    s^(1f0/emd.p)
end