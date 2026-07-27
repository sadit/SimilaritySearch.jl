# This file is a part of SimilaritySearch.jl
# export Cosine, Angle, NormCosine, NormAngle
# export L1, L2, SqL2, Lp, LInfty, Lp
using ..Dist: fastacos

"""
    Cosine()

Float32-casting variant of `Dist.Cosine`. The cosine distance is defined as

```math
1 - \\cos(u, v) = 1 - \\frac{\\sum_i u_i v_i}{\\sqrt{\\sum_i u_i^2}\\sqrt{\\sum_i v_i^2}}
```

Every element of `u` and `v` is cast to `Float32` before accumulating the dot product
and norms, regardless of the input element type. Access via `Dist.CastF32.Cosine`.

# Examples

```julia
d = Dist.CastF32.Cosine()
evaluate(d, u, v)
```
"""
struct Cosine <: SemiMetric end

"""
    Angle()

Float32-casting variant of `Dist.Angle`. Computes the angle between `u` and `v` as
``\\arccos(\\cos(u, v))``, casting every element of `u` and `v` to `Float32` before
accumulating the dot product and norms. Access via `Dist.CastF32.Angle`.

# Examples

```julia
d = Dist.CastF32.Angle()
evaluate(d, u, v)
```
"""
struct Angle <: SemiMetric end

"""
    NormCosine()

Float32-casting variant of `Dist.NormCosine`. Assumes that `u` and `v` are already
normalized, which reduces the cosine distance to ``1 - \\sum_i u_i v_i``, with every
element cast to `Float32` before accumulating the dot product. Access via
`Dist.CastF32.NormCosine`.

# Examples

```julia
d = Dist.CastF32.NormCosine()
evaluate(d, u, v)
```
"""
struct NormCosine <: SemiMetric end

"""
    NormAngle()

Float32-casting variant of `Dist.NormAngle`. Assumes that `u` and `v` are already
normalized, computing ``\\arccos\\left(\\sum_i u_i v_i\\right)``, with every element
cast to `Float32` before accumulating the dot product. Access via `Dist.CastF32.NormAngle`.

# Examples

```julia
d = Dist.CastF32.NormAngle()
evaluate(d, u, v)
```
"""
struct NormAngle <: SemiMetric end

@inline function dot32(a, b)::Float32
    d = 0.0f0
    
    @fastmath @inbounds @simd for i in eachindex(a, b)
        d = muladd(Float32(a[i]), Float32(b[i]), d)
    end

    d
end

@inline function norm32(a)::Float32
    sqrt(dot32(a, a))
end

@inline evaluate(::NormCosine, a, b) = 1.0f0 - dot32(a, b)
@inline evaluate(::NormAngle, a, b) = fastacos(dot32(a, b))
@inline evaluate(::Cosine, a, b) = 1.0f0 - dot32(a, b) / (norm32(a) * norm32(b))
@inline evaluate(::Angle, a, b) = fastacos(dot32(a, b) / (norm32(a) * norm32(b)))

"""
    L1()

Float32-casting variant of `Dist.L1`, the Manhattan or ``L_1`` distance

```math
L_1(u, v) = \\sum_i{|u_i - v_i|}
```

Every element of `u` and `v` is cast to `Float32` before accumulating the sum,
regardless of the input element type. Access via `Dist.CastF32.L1`.

# Examples

```julia
d = Dist.CastF32.L1()
evaluate(d, u, v)
```
"""
struct L1 <: SemiMetric end

@inline function evaluate(::L1, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        d += ifelse(m > 0, m, -m)
    end

    d
end

"""
    L2()

Float32-casting variant of `Dist.L2`, the euclidean or ``L_2`` distance

```math
L_2(u, v) = \\sqrt{\\sum_i{(u_i - v_i)^2}}
```

Every element of `u` and `v` is cast to `Float32` before accumulating the sum,
regardless of the input element type. Access via `Dist.CastF32.L2`.

# Examples

```julia
d = Dist.CastF32.L2()
evaluate(d, u, v)
```
"""
struct L2 <: SemiMetric end

@inline function evaluate(::L2, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        d += (Float32(a[i]) - Float32(b[i]))^2 #m * m
    end

    sqrt(d)
end

"""
    SqL2()

Float32-casting variant of `Dist.SqL2`, the squared euclidean distance

```math
L_2(u, v) = \\sum_i{(u_i - v_i)^2}
```

It avoids the computation of the square root, and every element of `u` and `v` is
cast to `Float32` before accumulating the sum, regardless of the input element type.
Access via `Dist.CastF32.SqL2`.

# Examples

```julia
d = Dist.CastF32.SqL2()
evaluate(d, u, v)
```
"""
struct SqL2 <: SemiMetric end

@inline function evaluate(::SqL2, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        d = muladd(m, m, d)
    end

    d
end

"""
    LInfty()

Float32-casting variant of `Dist.LInfty`, the Chebyshev or ``L_{\\infty}`` distance

```math
L_{\\infty}(u, v) = \\max_i{\\left| u_i - v_i \\right|}
```

Every element of `u` and `v` is cast to `Float32` before computing the maximum,
regardless of the input element type. Access via `Dist.CastF32.LInfty`.

# Examples

```julia
d = Dist.CastF32.LInfty()
evaluate(d, u, v)
```
"""
struct LInfty <: SemiMetric end

@inline function evaluate(::LInfty, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = abs(Float32(a[i]) - Float32(b[i]))
        d = max(d, m)
    end

    d
end

"""
    Lp(p)
    Lp(p, pinv)

Float32-casting variant of `Dist.Lp`, the general Minkowski ``L_p`` distance

```math
L_p(u, v) = \\left|\\sum_i{(u_i - v_i)^p}\\right|^{1/p}
```

where ``p_{inv} = 1/p`` (you may specify unrelated `p` and `pinv` if needed). Every
element of `u` and `v` is cast to `Float32` before accumulating the sum, regardless of
the input element type. Access via `Dist.CastF32.Lp`.

# Examples

```julia
d = Dist.CastF32.Lp(3f0)
evaluate(d, u, v)
```
"""
struct Lp <: SemiMetric
    p::Float32
    pinv::Float32
end

@inline Lp(p) = Lp(p, 1.0f0 / p)

@inline function evaluate(lp::Lp, a, b)
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = abs(Float32(a[i]) - Float32(b[i]))
        d += m^lp.p
    end

    d^lp.pinv
end
