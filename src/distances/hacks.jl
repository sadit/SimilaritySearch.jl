# This file is a part of SimilaritySearch.jl

export NegativeDistanceHack, SimilarityFromDistance, DistanceF32

"""
    NegativeDistanceHack(dist)

Evaluates as the negative of the distance function being wrapped.
This is not a real distance function but a simple hack to get a similarity and use it
for searching for farthest elements (farthest points / farthest pairs) on indexes that
can handle this hack (e.g., `ExhaustiveSearch`, `ParallelExhaustiveSearch`, `SearchGraph`).
"""
struct NegativeDistanceHack{Dist<:SemiMetric} <: SemiMetric
    dist::Dist
end

@inline evaluate(neg::NegativeDistanceHack, u, v) = -evaluate(neg.dist, u, v)

"""
    SimilarityFromDistance(dist)

Evaluates as ``1/(1 + d)`` for a distance evaluation ``d`` of `dist`.
This is not a distance function and is part of the hacks to get a similarity 
for searching farthest elements on indexes that can handle this hack
(e.g., `ExhaustiveSearch`, `ParallelExhaustiveSearch`, `SearchGraph`).
"""
struct SimilarityFromDistance{Dist<:SemiMetric} <: SemiMetric
    dist::Dist
end

@inline evaluate(sim::SimilarityFromDistance, u, v) = 1 / (1 + evaluate(sim.dist, u, v))

"""
    DistanceWithIdentifiers(distance, database)

Wraps the given database and distance with a proxy database that is accessed with integers from 1 to n
"""
struct DistanceWithIdentifiers{Dist<:SemiMetric,DB} <: SemiMetric
    dist::Dist
    db::DB
end

@inline evaluate(D::DistanceWithIdentifiers, i::Integer, j::Integer) = evaluate(D.dist, D.db[i], D.db[j])

"""
    DistanceF32(dist)

Useful for vector distances and legacy hardware using Float32 as the fastest datatype for computing.
It uses temporary representations for input vectors to always use Float32 vectors for the wrapped distance function.
"""
struct DistanceF32{Dist<:SemiMetric} <: SemiMetric
    dist::Dist
    caches::Matrix{Float32}
end

DistanceF32(dist::SemiMetric, dim::Int) = DistanceF32(dist, Matrix{Float32}(undef, dim, 2Threads.nthreads()))

@inline evaluate(D::DistanceF32, u::AbstractVector{Float32}, v::AbstractVector{Float32}) = evaluate(D.dist, u, v)

@inline function evaluate(D::DistanceF32, u::AbstractVector{Float32}, v::AbstractVector{<:AbstractFloat})
    v̂ = view(D, :, 2Threads.threadid())
    v̂ .= v
    evaluate(D.dist, u, v̂)
end

@inline function evaluate(D::DistanceF32, u::AbstractVector{<:AbstractFloat}, v::AbstractVector{Float32})
    û = view(D, :, 2Threads.threadid())
    û .= u
    evaluate(D.dist, û, v)
end

@inline function evaluate(D::DistanceF32, u::AbstractVector{<:AbstractFloat}, v::AbstractVector{<:AbstractFloat})
    i = 2Threads.threadid()
    û = view(D, :, i)
    v̂ = view(D, :, i-1)
    û .= u
    v̂ .= v
    evaluate(D.dist, û, v̂)
end

