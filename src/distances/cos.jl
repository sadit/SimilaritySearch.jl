export cosine_similarity, cosine_distance, angle_distance

using LinearAlgebra
import LinearAlgebra: normalize, normalize!

function normalize(X::AbstractVector{S}) where S<:AbstractVector
    normalize.(X)
end

function normalize!(X::AbstractVector{S}) where S<:AbstractVector
    for x in X
        normalize!(x)
    end

    X
end

function normalize!(X::AbstractMatrix)
    for x in eachcol(X)
        normalize!(x)
    end
    
    X
end

"""
    cosine_distance(a, b)::Float64

Computes the cosine distance between two vectors, it expects normalized vectors (see [normalize!](@ref) method).
Please use angle_distance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
function cosine_distance(a, b)::Float64
    1 - dot(a, b)
end

const π_2 = π / 2
"""
    angle_distance(a, b)::Float64

Computes the angle  between two SparseVector objects (sparse vectors).
It supposes that all vectors are normalized (see `normalize!` function)

"""
function angle_distance(a, b)::Float64
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