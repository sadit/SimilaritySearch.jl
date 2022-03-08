# This file is a part of SimilaritySearch.jl

export JaccardDistance, DiceDistance, IntersectionDissimilarity, CosineDistanceSet
import Distances: evaluate
using Base.Order

"""
    JaccardDistance()

The Jaccard distance is defined as

```math
J(u, v) = \\frac{|u \\cap v|}{|u \\cup v|}
```
"""
struct JaccardDistance <: SemiMetric end

"""
    DiceDistance()

The Dice distance is defined as

```math
D(u, v) = \\frac{2 |u \\cap v|}{|u| + |v|}
```
"""
struct DiceDistance <: SemiMetric end

"""
    IntersectionDissimilarity()

The intersection dissimilarity uses the size of the intersection as a mesuare of similarity as follows:

```math
I(u, v) = 1 - \\frac{|u \\cap v|}{\\max \\{|u|, |v|\\}}
```
"""
struct IntersectionDissimilarity <: SemiMetric end

"""
    intersectionsize(a, b, o=Forward)

Computes the size the intersections of `a` and `b`, specified as ordered sequences.
"""
function intersectionsize(a, b, o=Forward)
    len_a::Int = length(a)
    len_b::Int = length(b)
    ia::Int = ib::Int = 1
    intersection_size::Int = 0
    @inbounds while ia <= len_a && ib <= len_b
        if lt(o, a[ia], b[ib])
            ia += 1
        elseif lt(o, b[ib], a[ia])
            ib += 1
        else
            ia += 1
            ib += 1
            intersection_size += 1
        end
    end

    intersection_size
end

"""
    unionsize(a, b, isize)

Computes the size of the union of `a` and `b` that have an intersection size `isize`
"""
function unionsize(a, b, isize)
    length(a) + length(b) - isize
end

"""
    evaluate(::JaccardDistance, a, b)

Computes the Jaccard's distance of `a` and `b` both sets specified as
sorted vectors.
"""
function evaluate(::JaccardDistance, a, b)
    isize = intersectionsize(a, b)
    1.0 - isize / unionsize(a, b, isize)
end

"""
    evaluate(::DiceDistance, a, b)

Computes the Dice's distance of `a` and `b` both sets specified as
sorted vectors.
"""
function evaluate(::DiceDistance, a, b)
    i = intersectionsize(a, b)
    1.0 - 2 * i / (length(a) + length(b))
end


"""
    evaluate(::IntersectionDissimilarity, a, b)

Uses the intersection as a distance function (non-metric)
"""
function evaluate(::IntersectionDissimilarity, a, b)
    i = intersectionsize(a, b)
    return 1.0 - i / max(length(a), length(b))
end

struct CosineDistanceSet <: SemiMetric end
"""
    evaluate(::CosineDistanceSet, a, b)

Computes the cosine distance where `a` and `b` are sorted lists of integers (emulating binary sparse vectores)
"""
function evaluate(::CosineDistanceSet, U, V)
    1 - intersectionsize(U, V) / (sqrt(length(U)) * sqrt(length(V)))
end