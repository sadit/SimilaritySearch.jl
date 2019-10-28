export normalize!, cosine_similarity, cosine_distance, angle_distance

using LinearAlgebra


"""
angle_distance

Computes the angle between two vectors, it expects normalized vectors (see normalize! method)
"""
function angle_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    m = max(-1.0, dot(a, b))
    acos(min(1.0, m))
end

"""
cosine_distance

Computes the cosine distance between two vectors, it expects normalized vectors (see normalize! method).
Please use angle_distance if you are expecting a metric function (instead cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
function cosine_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    1 - dot(a, b)
end
