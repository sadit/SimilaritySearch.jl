# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogSatNeighborhood

struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64
    near::KnnResult
    LogSatNeighborhood(base=1.1) = new(base, KnnResult(1))
end


function find_neighborhood(algo::LogSatNeighborhood, index::SearchGraph, item)
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    N = Int32[]
    near = algo.near
    @inbounds for p in search(index, item, k)
        pobj = index.db[p.id]
        empty!(near)
        push!(near, p.id, p.dist)
        for nearID in N
            d = evaluate(index.dist, index.db[nearID], pobj)
            push!(near, nearID, d)
        end

        if first(near).id == p.id
            push!(N, p.id)
        end
    end

    N
end
