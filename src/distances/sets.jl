# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export JaccardDistance, DiceDistance, IntersectionDissimilarity
import Distances: evaluate

"""
    JaccardDistance()

The Jaccard distance is defined as

```math
J(u, v) = \\frac{|u \\cap v|}{|u \\cup v|}
```
"""
struct JaccardDistance <: PreMetric end

"""
    DiceDistance()

The Dice distance is defined as

```math
D(u, v) = \\frac{2 |u \\cap v|}{|u| + |v|}
```
"""
struct DiceDistance <: PreMetric end

"""
    IntersectionDissimilarity()

The intersection dissimilarity uses the size of the intersection as a mesuare of similarity as follows:

```math
I(u, v) = 1 - \\frac{|u \\cap v|}{\\max \\{|u|, |v|\\}}
```
"""
struct IntersectionDissimilarity <: PreMetric end

StructTypes.StructType(::Type{JaccardDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{DiceDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{IntersectionDissimilarity}) = StructTypes.Struct()

"""
    union_intersection(a::T, b::T)

Computes both the size of the unions an the size the intersections of `a` and `b`;
specified as ordered sequences.
"""
function union_intersection(a::T, b::T) where {T <: AbstractVector}
    len_a::Int = length(a)
    len_b::Int = length(b)
    ia::Int = 1
    ib::Int = 1
    intersection_size::Int = 0
    c::Int = 0
    @inbounds while ia <= len_a && ib <= len_b
        c = cmp(a[ia], b[ib])
        if c == 0
            ia += 1
            ib += 1
            intersection_size += 1
        elseif c < 0
            ia += 1
        else
            ib += 1
        end
    end

    len_a + len_b - intersection_size, intersection_size
end


"""
    evaluate(::JaccardDistance, a, b)

Computes the Jaccard's distance of `a` and `b` both sets specified as
sorted vectors.
"""
function evaluate(::JaccardDistance, a, b)
    u, i = union_intersection(a, b)
    1.0 - i / u
end

"""
    evaluate(::DiceDistance, a, b)

Computes the Dice's distance of `a` and `b` both sets specified as
sorted vectors.
"""
function evaluate(::DiceDistance, a, b)
    u, i = union_intersection(a, b)
    1.0 - 2 * i / (length(a) + length(b))
end


"""
    evaluate(::IntersectionDissimilarity, a, b)

Uses the intersection as a distance function (non-metric)
"""
function evaluate(::IntersectionDissimilarity, a, b)
    u, i = union_intersection(a, b)

    return 1.0 - i / max(length(a), length(b))
end