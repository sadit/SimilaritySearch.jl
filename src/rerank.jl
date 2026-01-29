# This file is part of SimilaritySearch.jl

export rerank!

function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractVector{IdWeight})
    m = 0
    for i in eachindex(res)
        p = res[i]
        if p.id == 0
            break
        else
            m = i
            o = db[p.id]
            d = evaluate(dist, o, q)
            res[i] = IdWeight(p.id, d)
        end
    end

    sort!(view(res, 1:m), by=x -> x.weight)
    res
end

function rerank!(dist::PreMetric, db::AbstractDatabase, queries::AbstractDatabase, knns::AbstractMatrix{IdWeight})
    m = length(queries)
    minbatch = getminbatch(m, Threads.nthreads(), 0)
    Threads.@threads :static for j in 1:minbatch:m
        for i in j:min(m, j + minbatch - 1)
            res = view(knns, :, i)
            rerank!(dist, db, queries[i], res)
        end
    end

    knns
end

function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractKnn)
    rerank!(dist, db, q, viewitems(res))
end

