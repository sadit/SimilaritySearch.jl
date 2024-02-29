# This file is part of SimilaritySearch.jl
#

export hsp_queries


function hsp_should_push(sat_neighborhood::T, dfun::SemiMetric, db::AbstractDatabase, item, id, dist) where T
    #D2 = evaluate(item, db[linkID])
    #dist = evaluate(item, obj)
    @inbounds obj = db[id]
    @inbounds for linkID in sat_neighborhood
        d = evaluate(dfun, obj, db[linkID])
        #d < dist && return true
        d - dist < 0 && return false
    end

    true 
end

"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns, dists; ctx = SearchGraphContext())
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext())
    hsp_queries(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext())


Computes the half-space partition of the queries `Q` (possibly given as `knns`, `dists`)


## Optional keyword arguments
- `ctx::SearchGraphContext` search context (caches)
- `α::Float32` hyperbolic parameter, α=0
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns, dists; ctx = SearchGraphContext(), α::Float32=0f0)
    n = length(Q)
    hsp = Vector{KnnResult}(undef, n)
    Threads.@threads :static for i = 1:n
        idlist = @view knns[:, i]
        distlist = @view dists[:, i]
        res = getknnresult(length(idlist), ctx)
        q = Q[i]
        for (objID, d) in zip(idlist, distlist)
            objID == 0 && break
            if hsp_should_push(IdView(res), dist, X, q, objID, d)
                push_item!(res, objID, d)
            end
        end

        hsp[i] = KnnResult(copy(res.items), length(res))
    end

    hsp
end

function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext())
    idx = ExhaustiveSearch(; dist, db=X)
    knns, dists = searchbatch(idx, Q, k)
    hsp_queries(dist, X, Q, knns, dists; ctx)
end

function hsp_queries(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext())
    knns, dists = searchbatch(idx, ctx, Q, k)
    hsp_queries(distance(idx), database(idx), Q, knns, dists; ctx)
end

