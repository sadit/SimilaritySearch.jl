# This file is a part of SimilaritySearch.jl

export CosineDistance, AngleDistance, NormalizedCosineDistance, NormalizedAngleDistance, TurboNormalizedCosineDistance, TurboCosineDistance
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

"""
   AngleDistance()
   
The angle distance is defined as:
```math
∠(u, v)= \\arccos(\\cos(u, v))
```

"""
struct AngleDistance <: SemiMetric end

"""
    NormalizedCosineDistance()

Similar to [`CosineDistance`](@ref) but suppose that input vectors are already normalized, and therefore, reduced to simply one minus the dot product.

```math
1 - \\sum_i {u_i v_i}
```

"""
struct NormalizedCosineDistance <: SemiMetric end


"""
    NormalizedAngleDistance()

Similar to [`AngleDistance`](@ref) but suppose that input vectors are already normalized

```math
\\arccos \\sum_i {u_i v_i}
```

"""
struct NormalizedAngleDistance <: SemiMetric end


"""
    TurboNormalizedCosineDistance()

The `@turbo`ed implementation of [`NormalizedCosineDistance`](@ref),
see [`LoopVectorization`](https://github.com/JuliaSIMD/LoopVectorization.jl).
"""
struct TurboNormalizedCosineDistance <: SemiMetric end

"""
    TurboNormalizedCosineDistance()

The `@turbo`ed implementation of [`CosineDistance`](@ref),
"""
struct TurboCosineDistance <: SemiMetric end


const π_2 = π / 2

function fastacos(d)
    if d <= -1.0
        π
    elseif d >= 1.0
        0.0
    elseif d == 0  # turn around for zero vectors, in particular for denominator=0
        π_2
    else
        acos(d)
    end
end

"""
    evaluate(::NormalizedCosineDistance, a, b)
    evaluate(::TurboNormalizedCosineDistance, a, b)

Computes the cosine distance between two vectors, it expects normalized vectors.
Please use NormalizedAngleDistance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
evaluate(::NormalizedCosineDistance, a::T, b) where T = one(eltype(T)) - dot(a, b)

function evaluate(::TurboNormalizedCosineDistance, a, b)
    T = typeof(a[1])
    s = zero(T)

    @turbo thread=1 unroll=4 for i in eachindex(a)
        s = muladd(a[i], b[i], s)
    end

    one(T) - s
end

"""
    evaluate(::TurboCosineDistance, a, b)


Turbo'ed version of cosine distance
"""
function evaluate(::TurboCosineDistance, a, b)
    T = typeof(a[1])
    s = zero(T)

    @turbo thread=1 unroll=4 for i in eachindex(a)
        s = muladd(a[i], b[i], s)
    end

    one(T) - s / (norm(a) * norm(b))
end


"""
    evaluate(::AngleDistance, a, b)

Computes the angle  between twovectors. It supposes that all vectors are normalized

"""
evaluate(::NormalizedAngleDistance, a, b) = fastacos(dot(a, b))

"""
    evaluate(::CosineDistance, a, b)

Computes the cosine distance between two vectors.
Please use AngleDistance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
evaluate(::CosineDistance, a, b) = one(eltype(a)) - dot(a, b) / (norm(a) * norm(b))

"""
    evaluate(::AngleDistance, a, b)

Computes the angle  between twovectors.

"""
function evaluate(::AngleDistance, a, b)
    d = dot(a, b) / (norm(a) * norm(b))
    fastacos(d)
end
