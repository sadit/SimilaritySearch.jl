# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export FixedNeighborhood

struct FixedNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function FixedNeighborhood()
    return FixedNeighborhood(8)
end

function optimize_neighborhood!(algo::FixedNeighborhood, index::SearchGraph{T}, dist, perf, recall) where T
end

"""
    find_neighborhood(algo::FixedNeighborhood, index::SearchGraph, item)

Finds a list of neighbors using the `FixedNeighborhood` criterion of item in the index
"""
function find_neighborhood(algo::FixedNeighborhood, index::SearchGraph, item)
    [p.id for p in search(index, item, algo.k)]
end
