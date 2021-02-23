# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export CosineDistance, AngleDistance, NormalizedCosineDistance, NormalizedAngleDistance
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
struct CosineDistance <: PreMetric end

"""
   AngleDistance()
   
The angle distance is defined as:
```math
∠(u, v)= \\arccos(\\cos(u, v))
```

"""
struct AngleDistance <: PreMetric end
"""
    NormalizedCosineDistance()

Similar to [`CosineDistance`](@ref) but suppose that input vectors are already normalized

```math
1 - \\sum_i {u_i v_i}
```

"""
struct NormalizedCosineDistance <: PreMetric end

"""
    NormalizedAngleDistance()

Similar to [`AngleDistance`](@ref) but suppose that input vectors are already normalized

```math
\\arccos \\sum_i {u_i v_i}
```

"""
struct NormalizedAngleDistance <: PreMetric end

StructTypes.StructType(::Type{CosineDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{AngleDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{NormalizedCosineDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{NormalizedAngleDistance}) = StructTypes.Struct()

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

Computes the cosine distance between two vectors, it expects normalized vectors (see [normalize!](@ref) method).
Please use NormalizedAngleDistance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
function evaluate(::NormalizedCosineDistance, a, b)
    one(eltype(a)) - dot(a, b)
end

"""
    evaluate(::AngleDistance, a, b)

Computes the angle  between twovectors. It supposes that all vectors are normalized (see `normalize!` function)

"""
function evaluate(::NormalizedAngleDistance, a, b)
    fastacos(dot(a, b))
end

"""
    evaluate(::CosineDistance, a, b)

Computes the cosine distance between two vectors.
Please use AngleDistance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
function evaluate(::CosineDistance, a, b)
    one(eltype(a)) - dot(a, b) / (norm(a) * norm(b))
end

"""
    evaluate(::AngleDistance, a, b)

Computes the angle  between twovectors.

"""
function evaluate(::AngleDistance, a, b)
    d = dot(a, b) / (norm(a) * norm(b))
    fastacos(d)
end
