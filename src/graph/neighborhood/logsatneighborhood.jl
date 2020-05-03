# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogSatNeighborhood

struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64

    LogSatNeighborhood(base=1.1) = new(base)
end

function optimize_neighborhood!(algo::LogSatNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where {T}
    # optimize_neighborhood!(algo.g, index, perf, recall)
end

function fix_neighborhood!(index::SearchGraph, dist::Function)
    if !(index.neighborhood_algo isa LogSatNeighborhood)
        return
    end
    knn = KnnResult(1)
    for i in eachindex(index.db)
        fix_neighborhood!(index, dist, i, knn)
    end
end

function fix_neighborhood!(index::SearchGraph, dist::Function, objID::Integer, knn::KnnResult)
    n = length(index.db)
    k = n
    reset!(knn, k)
    obj = index.db[objID]
    for link in index.links[objID]
        push!(knn, link, dist(obj, index.db[link]))
    end
    # reverse!(knn.pool)

    near = KnnResult(1)
    
    N = index.links[objID]
    empty!(N)

    @inbounds for (i, p) in enumerate(knn.pool)
        #dqp = p.dist
        obj = index.db[p.objID]
        empty!(near)
        push!(near, p.objID, p.dist)
        for nearID in N
            d = convert(Float32, dist(index.db[nearID], obj))
            push!(near, nearID, d)
        end

        if first(near).objID == p.objID
            push!(N, p.objID)
        end
    end

end

function neighborhood(algo::LogSatNeighborhood, index::SearchGraph{T}, dist::Function, item::T, knn::KnnResult, N::Vector, searchctx) where {T}
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    reset!(knn, k)
    empty!(N)
    search(index, dist, item, knn; searchctx=searchctx)

    near = KnnResult(1)
    # reverse!(knn.pool)
    @inbounds for p in knn.pool
        pobj = index.db[p.objID]
        empty!(near)
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
end
