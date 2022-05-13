# This file is a part of SimilaritySearch.jl

export NegativeDistanceHack, SimilarityFromDistance

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