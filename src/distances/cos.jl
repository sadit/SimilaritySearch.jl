export cosine_similarity, cosine_distance, angle_distance

using LinearAlgebra

"""
cosine_distance

Computes the cosine distance between two vectors, it expects normalized vectors (see [normalize!](@ref) method).
Please use angle_distance if you are expecting a metric function (instead cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
function cosine_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    1 - dot(a, b)
end


const π_2 = π / 2
"""
angle_distance

Computes the angle  between two SparseVector objects (sparse vectors).

It supposes that all vectors are normalized (see `normalize!` function)

"""
function angle_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    d = dot(a, b)

    if d <= -1.0
        return π
    elseif d >= 1.0
        return 0.0
    elseif d == 0  # turn around for zero vectors, in particular for denominator=0
        return π_2
    else
        return acos(d)
    end
end