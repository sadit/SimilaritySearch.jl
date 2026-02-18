# This file is a part of SimilaritySearch.jl
export L1, L2, SqL2, Lp, LInfty

###################
#
#  L1 - Manhattan distance
#
#################
"""
    L1()

The manhattan distance or ``L_1`` is defined as

```math
L_1(u, v) = \\sum_i{|u_i - v_i|}
```

"""
struct L1 <: Metric end

"""
    evaluate(::L1, a, b)

Computes the Manhattan's distance between `a` and `b`
"""
@inline function evaluate(::L1, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = a[i] - b[i]
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
    L2()

The euclidean distance or ``L_2`` is defined as

```math
L_2(u, v) = \\sqrt{\\sum_i{(u_i - v_i)^2}}
```
"""
struct L2 <: Metric end

"""
    evaluate(::L2, a, b)
    
Computes the Euclidean's distance betweem `a` and `b`
"""
@inline function evaluate(::L2, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a, b)
        d += (a[i] - b[i])^2 #m * m
    end

    sqrt(d)
end

###############
#
#  Squared L2 dissimilarity
# 
###############
"""
    SqL2()

The squared euclidean distance is defined as

```math
L_2(u, v) = \\sum_i{(u_i - v_i)^2}
```

It avoids the computation of the square root and should be used
whenever you are able do it.
"""
struct SqL2 <: SemiMetric end

"""
    evaluate(::SqL2, a, b)

Computes the squared Euclidean's distance between `a` and `b`
"""
@inline function evaluate(::SqL2, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = a[i] - b[i]
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
    LInfty()

The Chebyshev or ``L_{\\infty}`` distance is defined as

```math
L_{\\infty}(u, v) = \\max_i{\\left| u_i - v_i \\right|}
```

"""
struct LInfty <: Metric end

"""
    evaluate(::LInfty, a, b)

Computes the maximum distance or Chebyshev's distance
"""
@inline function evaluate(::LInfty, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = abs(a[i] - b[i])
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
    Lp(p)
    Lp(p, pinv)

The general Minkowski distance ``L_p`` distance is defined as

```math
L_p(u, v) = \\left|\\sum_i{(u_i - v_i)^p}\\right|^{1/p}
```

Where ``p_{inv} = 1/p``. Note that you can specify unrelated `p` and `pinv` if you need an specific behaviour.
"""
struct Lp <: SemiMetric
    p::Float32
    pinv::Float32
end

@inline Lp(p) = Lp(p, 1.0f0 / p)

"""
    evaluate(lp::Lp, a, b)

Computes generic Minkowski's distance
"""
@inline function evaluate(lp::Lp, a, b)
    d = zero(eltype(a))

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = abs(a[i] - b[i])
        d += m^lp.p
    end

    d^lp.pinv
end
