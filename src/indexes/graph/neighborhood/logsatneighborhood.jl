export LogSatNeighborhood

struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function LogSatNeighborhood()
    return LogSatNeighborhood(1.1)
end

function optimize_neighborhood!{T}(algo::LogSatNeighborhood, index::LocalSearchIndex{T}, perf, recall)
    # optimize_neighborhood!(algo.g, index, perf, recall)
end

function neighborhood{T}(algo::LogSatNeighborhood, index::LocalSearchIndex{T}, item::T)
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    knn = search(index, item, KnnResult(k))
    N = Vector{Int32}(0)

    @inbounds for p in knn
        dqp = p.dist
        pobj = index.db[p.objID]
        near = NnResult()
        push!(near, p.objID, p.dist)
        for nearID in N
            d = convert(Float32, index.dist(index.db[nearID], pobj)) 
            push!(near, nearID, d)
        end

        f = first(near)
        if f.objID == p.objID
            push!(N, p.objID)
        end
    end

    return knn, N
end
