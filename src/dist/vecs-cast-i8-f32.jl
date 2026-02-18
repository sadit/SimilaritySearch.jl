# This file is a part of SimilaritySearch.jl
export Cosine, Angle, NormCosine, NormAngle
export L1, L2, SqL2, Lp, LInfty, Lp
using ..Dist: fastacos

struct Cosine <: SemiMetric end
struct Angle <: SemiMetric end
struct NormCosine <: SemiMetric end
struct NormAngle <: SemiMetric end

### The core idea of this module specialization is to avoid overflows in high dimensional vectors (i.e., accumulated squared differences like 127^2)
const SCALE = 1f0/128f0

@inline function dot(a, b)::Float32
    d = 0.0f0

    @fastmath @inbounds @simd for i in eachindex(a, b)
        d = muladd(Float32(a[i]) * SCALE, Float32(b[i]) * SCALE, d)
    end

    d
end

@inline function norm(a)::Float32
    sqrt(dot(a, a))
end

@inline evaluate(::NormCosine, a, b)::Float32 = 1.0f0 - dot(a, b)
@inline evaluate(::NormAngle, a, b)::Float32 = acos(dot(a, b))
@inline evaluate(::Cosine, a, b)::Float32 = 1.0f0 - dot(a, b) / (norm(a) * norm(b))
@inline evaluate(::Angle, a, b)::Float32 = acos(dot(a, b) / (norm(a) * norm(b)))

"""
    L1()

"""
struct L1 <: SemiMetric end

@inline function evaluate(::L1, a, b)::Float32
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        m *= SCALE
        d += ifelse(m > 0, m, -m)
    end

    d
end

"""
    L2()

"""
struct L2 <: SemiMetric end

@inline function evaluate(::L2, a, b)::Float32
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        m *= SCALE
        d += m * m
    end

    sqrt(d)
end

"""
    SqL2()

"""
struct SqL2 <: SemiMetric end

@inline function evaluate(::SqL2, a, b)::Float32
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        m *= SCALE
        d = muladd(m, m, d)
    end

    d
end

@inline function evaluate(::SqL2, a::AbstractVector{Int8}, b::AbstractVector{Int8})
    d = zero(Float32)
    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = Float32(a[i]) - Float32(b[i])
        m *= SCALE
        d = muladd(m, m, d)
    end

    d
end


"""
    LInfty()

"""
struct LInfty <: SemiMetric end

@inline function evaluate(::LInfty, a, b)::Float32
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = abs(Float32(a[i]) - Float32(b[i]))
        m *= SCALE
        d = max(d, m)
    end

    d
end

"""
    Lp(p)
    Lp(p, pinv)

"""
struct Lp <: SemiMetric
    p::Float32
    pinv::Float32
end

@inline Lp(p) = Lp(p, 1.0f0 / p)

@inline function evaluate(lp::Lp, a, b)::Float32
    d = zero(Float32)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = abs(Float32(a[i]) - Float32(b[i]))
        m *= SCALE
        d += m^lp.p
    end

    d^lp.pinv
end
