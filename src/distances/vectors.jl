# This file is a part of SimilaritySearch.jl
export L1Distance, L2Distance, SqL2Distance, LpDistance, LInftyDistance
export L1_asf32, L2_asf32, SqL2_asf32, Lp_asf32, LInfty_asf32
import Distances: evaluate

###################
#
#  L1 - Manhattan distance
#
#################
"""
    L1Distance()

The manhattan distance or ``L_1`` is defined as

```math
L_1(u, v) = \\sum_i{|u_i - v_i|}
```

"""
struct L1Distance <: SemiMetric end

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
    L1_asf32()

"""
struct L1_asf32 <: SemiMetric end

function evaluate(::L1_asf32, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        d += ifelse(m > 0, m, -m)
    end

    d
end

###################
#
# L2/Euclidean distance
#
##################

"""
    L2Distance()

The euclidean distance or ``L_2`` is defined as

```math
L_2(u, v) = \\sqrt{\\sum_i{(u_i - v_i)^2}}
```
"""
struct L2Distance <: SemiMetric end

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
    L2_asf32()

"""
struct L2_asf32 <: SemiMetric end

function evaluate(::L2_asf32, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a)
        d += (Float32(a[i]) - Float32(b[i]))^2 #m * m
    end

    sqrt(d)
end

###############
#
#  Squared L2 dissimilarity
# 
###############
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
    evaluate(::SqL2Distance, a, b)

Computes the squared Euclidean's distance between `a` and `b`
"""
function evaluate(::SqL2Distance, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a)
        m = a[i] - b[i]
        d = muladd(m, m, d)
    end

    d
end

"""
    SqL2_asf32()

"""
struct SqL2_asf32 <: SemiMetric end

function evaluate(::SqL2_asf32, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a)
        m = Float32(a[i]) - Float32(b[i])
        d = muladd(m, m, d)
    end

    d
end

###############
#
# Chebyshev/L∞/Lmax distance
#
###############
"""
    LInftyDistance()

The Chebyshev or ``L_{\\infty}`` distance is defined as

```math
L_{\\infty}(u, v) = \\max_i{\\left| u_i - v_i \\right|}
```

"""
struct LInftyDistance <: SemiMetric end

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
    LInfty_asf32()

"""
struct LInfty_asf32 <: SemiMetric end

function evaluate(::LInfty_asf32, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a)
        m = abs(Float32(a[i]) - Float32(b[i]))
        d = max(d, m)
    end

    d
end

##############
#
# Minkowski/Lp distance function family
#
##############

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

"""
    Lp_asf32(p)
    Lp_asf32(p, pinv)

"""
struct Lp_asf32 <: SemiMetric
    p::Float32
    pinv::Float32
end

Lp_asf32(p) = Lp_asf32(p, 1f0/p)

function evaluate(lp::Lp_asf32, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a)
        m = abs(Float32(a[i]) - Float32(b[i]))
        d += m ^ lp.p
    end

    d ^ lp.pinv
end

