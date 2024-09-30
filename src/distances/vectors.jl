# This file is a part of SimilaritySearch.jl
using LoopVectorization
export L1Distance, L2Distance, SqL2Distance, LpDistance, LInftyDistance
import Distances: evaluate

"""
    L1Distance()

The manhattan distance or ``L_1`` is defined as

```math
L_1(u, v) = \\sum_i{|u_i - v_i|}
```

"""
struct L1Distance <: SemiMetric end

"""
    L2Distance()

The euclidean distance or ``L_2`` is defined as

```math
L_2(u, v) = \\sqrt{\\sum_i{(u_i - v_i)^2}}
```
"""
struct L2Distance <: SemiMetric end

"""
    SqL2Distance()

The squared euclidean distance is defined as

```math
L_2(u, v) = \\sum_i{(u_i - v_i)^2}
```

It avoids the computation of the square root and should be used
whenever you are able do it.
"""
struct SqL2Distance <: SemiMetric end

"""
    LInftyDistance()

The Chebyshev or ``L_{\\infty}`` distance is defined as

```math
L_{\\infty}(u, v) = \\max_i{\\left| u_i - v_i \\right|}
```

"""
struct LInftyDistance <: SemiMetric end

"""
    LpDistance(p)
    LpDistance(p, pinv)

The general Minkowski distance ``L_p`` distance is defined as

```math
L_p(u, v) = \\left|\\sum_i{(u_i - v_i)^p}\\right|^{1/p}
```

Where ``p_{inv} = 1/p``. Note that you can specify unrelated `p` and `pinv` if you need an specific behaviour.
"""
struct LpDistance <: SemiMetric
    p::Float32
    pinv::Float32
end

LpDistance(p) = LpDistance(p, 1f0/p)

###################

"""
    evaluate(::L1Distance, a, b)

Computes the Manhattan's distance between `a` and `b`
"""
function evaluate(::L1Distance, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a, b)
	    m = a[i] - b[i]
        d += ifelse(m > 0, m, -m)
    end

    d
end

"""
    evaluate(::L2Distance, a, b)
    
Computes the Euclidean's distance betweem `a` and `b`
"""
function evaluate(::L2Distance, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a)
        d += (a[i] - b[i])^2 #m * m
    end

    sqrt(d)
end

"""
    evaluate(::SqL2Distance, a, b)

Computes the squared Euclidean's distance between `a` and `b`
"""
function evaluate(::SqL2Distance, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a)
        m = a[i] - b[i]
        d += m^2
    end

    d
end

"""
    evaluate(::LInftyDistance, a, b)

Computes the maximum distance or Chebyshev's distance
"""
function evaluate(::LInftyDistance, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a)
        m = abs(a[i] - b[i])
        d = max(d, m)
    end

    d
end

"""
    evaluate(lp::LpDistance, a, b)

Computes generic Minkowski's distance
"""
function evaluate(lp::LpDistance, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a)
        m = abs(a[i] - b[i])
        d += m ^ lp.p
    end

    d ^ lp.pinv
end
