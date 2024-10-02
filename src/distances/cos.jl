# This file is a part of SimilaritySearch.jl

export CosineDistance, AngleDistance, NormalizedCosineDistance, NormalizedAngleDistance
export Cosine_asf32, Angle_asf32, NormalizedCosine_asf32, NormalizedAngle_asf32
using LinearAlgebra
import Distances: evaluate

"""
   CosineDistance()
   
The cosine is defined as:
```math
\\cos(u, v) = \\frac{\\sum_i u_i v_i}{\\sqrt{\\sum_i u_i^2} \\sqrt{\\sum_i v_i^2}}
```

The cosine distance is defined as ``1 - \\cos(u,v)``
"""
struct CosineDistance <: SemiMetric end
struct Cosine_asf32 <: SemiMetric end

"""
   AngleDistance()
   
The angle distance is defined as:
```math
∠(u, v)= \\arccos(\\cos(u, v))
```

"""
struct AngleDistance <: SemiMetric end
struct Angle_asf32 <: SemiMetric end

"""
    NormalizedCosineDistance()

Similar to [`CosineDistance`](@ref) but suppose that input vectors are already normalized, and therefore, reduced to simply one minus the dot product.

```math
1 - \\sum_i {u_i v_i}
```

"""
struct NormalizedCosineDistance <: SemiMetric end
struct NormalizedCosine_asf32 <: SemiMetric end


"""
    NormalizedAngleDistance()

Similar to [`AngleDistance`](@ref) but suppose that input vectors are already normalized

```math
\\arccos \\sum_i {u_i v_i}
```

"""
struct NormalizedAngleDistance <: SemiMetric end
struct NormalizedAngle_asf32 <: SemiMetric end



const π_2 = Float32(π / 2)

function fastacos(d::Float32)
    if d <= -1f0
        π
    elseif d >= 1f0
        0.0
    elseif d == 0f0  # turn around for zero vectors, in particular for denominator=0
        π_2
    else
        acos(d)
    end
end

"""
    evaluate(::NormalizedCosineDistance, a, b)

Computes the cosine distance between two vectors, it expects normalized vectors.
Please use NormalizedAngleDistance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
evaluate(::NormalizedCosineDistance, a::T, b) where T = one(eltype(T)) - dot(a, b)

function evaluate(::NormalizedCosineDistance, a::AbstractVector{Float32}, b::AbstractVector{Float32})
    d = 0f0
    @fastmath @inbounds @simd for i in eachindex(a, b)
        d = muladd(a[i], b[i], d)
    end

    1f0 - d
end

function dot_asf32(a, b)
    d = 0f0
    @fastmath @inbounds @simd for i in eachindex(a, b)
        d = muladd(Float32(a[i]), Float32(b[i]), d)
    end

    d
end

function norm_asf32(a)
    d = 0f0
    @fastmath @inbounds @simd for i in eachindex(a)
        d = muladd(Float32(a[i]), Float32(a[i]), d)
    end

    sqrt(d)
end

evaluate(::NormalizedCosine_asf32, a, b) = 1f0 - dot_asf32(a, b)

"""
    evaluate(::AngleDistance, a, b)

Computes the angle  between twovectors. It supposes that all vectors are normalized

"""
evaluate(::NormalizedAngleDistance, a, b) = fastacos(dot(a, b))
evaluate(::NormalizedAngle_asf32, a, b) = fastacos(dot_asf32(a, b))

"""
    evaluate(::CosineDistance, a, b)

Computes the cosine distance between two vectors.
Please use AngleDistance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
evaluate(::CosineDistance, a, b) = one(eltype(a)) - dot(a, b) / (norm(a) * norm(b))
evaluate(::Cosine_asf32, a, b) = 1f0 - dot_asf32(a, b) / (norm_asf32(a) * norm_asf32(b))

"""
    evaluate(::AngleDistance, a, b)

Computes the angle  between twovectors.

"""
evaluate(::AngleDistance, a, b) = fastacos(dot(a, b) / (norm(a) * norm(b)))
evaluate(::Angle_asf32, a, b) = fastacos(dot_asf32(a, b) / (norm_asf32(a) * norm_asf32(b)))
