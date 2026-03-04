# This file is part of SimilaritySearch.jl

export rerank!

function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractVector{IdDist})
    m = 0
    for i in eachindex(res)
        p = res[i]
        if p.id == 0
            break
        else
            m = i
            o = db[p.id]
            d = evaluate(dist, o, q)
            res[i] = IdDist(p.id, d)
        end
    end

    sort!(view(res, 1:m), by=x -> x.dist)
    res
end

function rerank!(dist::PreMetric, db::AbstractDatabase, queries::AbstractDatabase, knns::AbstractMatrix{IdDist})
    m = length(queries)
    minbatch = getminbatch(m, Threads.nthreads(), 0)
    @batch per=core minbatch=minbatch for i in 1:m
        res = view(knns, :, i)
        rerank!(dist, db, queries[i], res)
    end

    knns
end

function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractKnn)
    rerank!(dist, db, q, viewitems(res))
end

