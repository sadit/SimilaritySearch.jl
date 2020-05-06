# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export jaccard_distance, dice_distance, union_intersection, intersection_distance

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

    return len_a + len_b - intersection_size, intersection_size
end

"""
    jaccard_distance(a, b)

Computes the Jaccard's distance of `a` and `b` both sets specified as
sorted vectors.
"""

function jaccard_distance(a, b)
    u, i = union_intersection(a, b)
    return 1.0 - i / u
end

"""
    dice_distance(a, b)

Computes the Dice's distance of `a` and `b` both sets specified as
sorted vectors.
"""
function dice_distance(a, b)
    u, i = union_intersection(a, b)
    return 1.0 - 2 * i / (length(a) + length(b))
end

"""
    intersection_distance(a, b)
(a, b)

Uses the intersection as a distance function (non-metric)
"""
function intersection_distance(a, b)
    u, i = union_intersection(a, b)

    return 1.0 - i / min(length(a), length(b))
end