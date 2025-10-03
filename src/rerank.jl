# This file is part of SimilaritySearch.jl

export rerank!

function rerank!(dist::SemiMetric, db::AbstractDatabase, q, res::AbstractVector{IdWeight})
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

function rerank!(dist::SemiMetric, db::AbstractDatabase, queries::AbstractDatabase, knns::AbstractMatrix{IdWeight})
    @batch minbatch = 4 per = thread for qID in eachindex(queries)
        res = view(knns, :, qID)
        rerank!(dist, db, queries[qID], res)
    end

    knns
end

function rerank!(dist::SemiMetric, db::AbstractDatabase, q, res::AbstractKnn)
    rerank!(dist, db, q, viewitems(res))
end

