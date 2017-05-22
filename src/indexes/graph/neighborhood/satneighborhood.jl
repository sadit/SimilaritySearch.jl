export SatNeighborhood

struct SatNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function SatNeighborhood()
    return SatNeighborhood(64)
end

function optimize_neighborhood!{T}(algo::SatNeighborhood, index::LocalSearchIndex{T}, perf, recall)
end

function neighborhood{T}(algo::SatNeighborhood, index::LocalSearchIndex{T}, item::T)
    N = Int32[]
    knn = search(index, item, KnnResult(algo.k))
    @inbounds for p in knn
        pobj = index.db[p.objID]
        near = NnResult()
        push!(near, zero(Int32), p.dist)
        for nearID in N
            d = convert(Float32, index.dist(index.db[nearID], pobj)) 
            push!(near, nearID, d)
        end

        if first(near).objID == 0
            push!(N, p.objID)
        end
    end

    return knn, N
end
