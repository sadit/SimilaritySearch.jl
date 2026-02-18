# This file is a part of SimilaritySearch.jl

export Cosine, Angle, NormCosine, NormAngle
using LinearAlgebra

"""
   Cosine()
   
The cosine is defined as:
```math
\\cos(u, v) = \\frac{\\sum_i u_i v_i}{\\sqrt{\\sum_i u_i^2} \\sqrt{\\sum_i v_i^2}}
```

The cosine distance is defined as ``1 - \\cos(u,v)``
"""
struct Cosine <: SemiMetric end

"""
   Angle()
   
The angle distance is defined as:
```math
∠(u, v)= \\arccos(\\cos(u, v))
```

"""
struct Angle <: Metric end

"""
    NormCosine()

Similar to [`Cosine`](@ref) but suppose that input vectors are already normalized, and therefore, reduced to simply one minus the dot product.

```math
1 - \\sum_i {u_i v_i}
```

"""
struct NormCosine <: SemiMetric end


"""
    NormAngle()

Similar to [`Angle`](@ref) but suppose that input vectors are already normalized

```math
\\arccos \\sum_i {u_i v_i}
```

"""
struct NormAngle <: Metric end

const π_2 = Float32(π / 2)

@inline fastacos(d::AbstractFloat) = fastacos(convert(Float32, d))
@inline function fastacos(d::Float32)::Float32
    if d <= -1.0f0
        π
    elseif d >= 1.0f0
        0.0f0
    elseif d == 0.0f0  # turn around for zero vectors, in particular for denominator=0
        π_2
    else
        acos(d)
    end
end

"""
    evaluate(::NormCosine, a, b)

Computes the cosine distance between two vectors, it expects normalized vectors.
Please use NormAngle if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
@inline evaluate(::NormCosine, a::T, b) where {T} = one(eltype(T)) - dot(a, b)

@inline function evaluate(::NormCosine, a::AbstractVector{Float32}, b::AbstractVector{Float32})
    d = 0.0f0
    @fastmath @inbounds @simd for i in eachindex(a, b)
        d = muladd(a[i], b[i], d)
    end

    1.0f0 - d
end


"""
    evaluate(::Angle, a, b)

Computes the angle  between twovectors. It supposes that all vectors are normalized

"""
@inline evaluate(::NormAngle, a, b) = fastacos(dot(a, b))

"""
    evaluate(::Cosine, a, b)

Computes the cosine distance between two vectors.
Please use Angle if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
@inline evaluate(::Cosine, a, b) = one(eltype(a)) - dot(a, b) / (norm(a) * norm(b))

"""
    evaluate(::Angle, a, b)

Computes the angle  between twovectors.

"""
@inline evaluate(::Angle, a, b) = fastacos(dot(a, b) / (norm(a) * norm(b)))
