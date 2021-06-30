# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogSatNeighborhood

"""
    LogSatNeighborhood(b=1.1)

New items are connected with a half partitioning scheme based on the Spatial access tree (**cite**),
``\\log_b n`` near items are considered for this procedure. The Log-Sat method will produce small neighborhood with
a nice covering of the search space.
"""
struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64
    near::KnnResult
end

LogSatNeighborhood(base=1.1) = LogSatNeighborhood(base, KnnResult(1))

Base.copy(algo::LogSatNeighborhood) = LogSatNeighborhood(algo.base)

function find_neighborhood(algo::LogSatNeighborhood, index::SearchGraph, item)
    n = length(index)
    k = max(1, ceil(Int, log(algo.base, n)))
    N = Int32[]
    near = algo.near

    @inbounds for (id, dist) in search(index, item, k)
        pobj = index[id]
        empty!(near)
        push!(near, 0, dist)
        for nearID in N
            d = evaluate(index.dist, index[nearID], pobj)
            push!(near, nearID, d)
        end

        if first(near.id) == 0
            push!(N, id)
        end
    end

    N
end
