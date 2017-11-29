export JaccardDistance, DiceDistance, union_intersection, IntersectionDistance

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
sim_jaccard computes the Jaccard's coefficient between two sets
represented as sorted arrays.
Note: 1.0 - sim_jaccard computes the Jaccard's distance
"""

mutable struct JaccardDistance
    calls::Int
    JaccardDistance() = new(0)
end

function (o::JaccardDistance)(a::T, b::T)::Float64 where {T <: AbstractVector}
    o.calls += 1
    u, i = union_intersection(a, b)
    return 1.0 - i / u
end

mutable struct DiceDistance
    calls::Int
    DiceDistance() = new(0)
end

function (o::DiceDistance)(a::T, b::T)::Float64 where {T <: AbstractVector}
    o.calls += 1
    u, i = union_intersection(a, b)
    return 1.0 - 2 * i / (length(a) + length(b))
end

mutable struct IntersectionDistance
    calls::Int
    IntersectionDistance() = new(0)
end

function (o::IntersectionDistance)(a::T, b::T)::Float64 where {T <: AbstractVector}
    o.calls += 1
    u, i = union_intersection(a, b)

    return 1.0 - i / min(length(a), length(b))
end
