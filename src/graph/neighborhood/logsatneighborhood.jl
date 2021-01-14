# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogSatNeighborhood

struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64

    LogSatNeighborhood(base=1.1) = new(base)
end

function optimize_neighborhood!(algo::LogSatNeighborhood, index::SearchGraph{T}, dist, perf, recall) where {T}
    # optimize_neighborhood!(algo.g, index, perf, recall)
end

function fix_neighborhood!(index::SearchGraph, dist)
    if !(index.neighborhood_algo isa LogSatNeighborhood)
        return
    end
    
    knn = KnnResult(1)
    for i in eachindex(index.db)
        fix_neighborhood!(index, dist, i, knn)
    end
end

function fix_neighborhood!(index::SearchGraph, dist, id::Integer, knn::KnnResult)
    n = length(index.db)
    k = n
    empty!(knn, k)
    obj = index.db[id]
    for link in index.links[id]
        push!(knn, link, dist(obj, index.db[link]))
    end
    # reverse!(knn.pool)

    near = KnnResult(1)
    
    N = index.links[id]
    empty!(N)

    @inbounds for (i, p) in enumerate(knn)
        #dqp = p.dist
        obj = index.db[p.id]
        empty!(near)
        push!(near, p.id, p.dist)
        for nearID in N
            d = dist(index.db[nearID], obj)
            push!(near, nearID, d)
        end

        if first(near).id == p.id
            push!(N, p.id)
        end
    end

end

function neighborhood(algo::LogSatNeighborhood, index::SearchGraph{T}, dist, item::T, knn::KnnResult, N::Vector, searchctx) where {T}
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    empty!(knn, k)
    empty!(N)
    search(index, dist, item, knn; searchctx=searchctx)

    near = KnnResult(1)
    @inbounds for p in knn
        pobj = index.db[p.id]
        empty!(near)
        push!(near, p.id, p.dist)
        for nearID in N
            d = dist(index.db[nearID], pobj)
            push!(near, nearID, d)
        end

        if first(near).id == p.id
            push!(N, p.id)
        end
    end
end
