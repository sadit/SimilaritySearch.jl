# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export cosine_distance, angle_distance, full_cosine_distance, full_angle_distance

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
    cosine_distance(a, b)

Computes the cosine distance between two vectors, it expects normalized vectors (see [normalize!](@ref) method).
Please use angle_distance if you are expecting a metric function (cosine_distance is a faster
alternative whenever the triangle inequality is not needed)
"""
function cosine_distance(a, b)
    1 - dot(a, b)
end

const π_2 = π / 2
"""
    angle_distance(a, b)

Computes the angle  between twovectors. It supposes that all vectors are normalized (see `normalize!` function)

"""
function angle_distance(a, b)
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


"""
    full_cosine_distance(a, b)

Computes the cosine similarity between two vectors.
"""

function full_cosine_similarity(a, b)
    dot(a, b) / norm(a) / norm(b)
end

"""
    full_cosine_distance(a, b)

Computes the cosine distance between two vectors.
Please use `full_angle_distance` if you are expecting a metric function (`full_cosine_distance` is a faster
alternative whenever the triangle inequality is not needed)
"""
function full_cosine_distance(a, b)
    1 - full_cosine_similarity(a, b)
end


"""
    full_angle_distance(a, b)

Computes the angle between two vectors.

"""
function full_angle_distance(a, b)
    d = full_cosine_similarity(a, b)

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

