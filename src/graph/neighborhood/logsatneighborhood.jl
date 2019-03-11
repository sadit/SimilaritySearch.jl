export LogSatNeighborhood

struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function LogSatNeighborhood()
    return LogSatNeighborhood(1.1)
end

function optimize_neighborhood!(algo::LogSatNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where {T}
    # optimize_neighborhood!(algo.g, index, perf, recall)
end

function neighborhood(algo::LogSatNeighborhood, index::SearchGraph{T}, dist::Function, item::T) where {T}
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    knn = search(index, dist, item, KnnResult(k))
    N = Vector{Int32}(undef, 0)

    @inbounds for p in knn
        dqp = p.dist
        pobj = index.db[p.objID]
        near = NnResult()
        push!(near, p.objID, p.dist)
        for nearID in N
            d = convert(Float32, dist(index.db[nearID], pobj)) 
            push!(near, nearID, d)
        end

        f = first(near)
        if f.objID == p.objID
            push!(N, p.objID)
        end
    end

    return knn, N
end
