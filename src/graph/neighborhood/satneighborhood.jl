# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export SatNeighborhood

"""
    SatNeighborhood(k=32)

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are reduced to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodAlgorithm
    k::Int
    near::KnnResult
    SatNeighborhood(k=32, res=KnnResult(1)) = new(k, res)
end

StructTypes.StructType(::Type{SatNeighborhood}) = StructTypes.Struct()
Base.copy(s::SatNeighborhood) = SatNeighborhood(s.k, KnnResult(1))

function find_neighborhood(algo::SatNeighborhood, index::SearchGraph, item)
    near = algo.near
    N = Int32[]
    @inbounds for p in search(index, item, algo.k)
        pobj = index.db[p.id]
        empty!(near)
        push!(near, zero(Int32), p.dist)
        for nearID in N
            d = evaluate(index.dist, index.db[nearID], pobj)
            push!(near, nearID, d)
        end

        if first(near).id == 0
            push!(N, p.id)
        end
    end

    N
end
