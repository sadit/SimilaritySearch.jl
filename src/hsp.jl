# This file is part of SimilaritySearch.jl
#

export hsp_queries

"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns, dists; ctx = SearchGraphContext())
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext())
    hsp_queries(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext())


Computes the half-space partition of the queries `Q` (possibly given as `knns`, `dists`)

"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns, dists; ctx = SearchGraphContext())
    n = length(Q)
    hsp = Vector{KnnResult}(undef, n)
    Threads.@threads :static for i = 1:n
        idlist = @view knns[:, i]
        distlist = @view dists[:, i]
        res = getknnresult(length(idlist), ctx)
        near = SimilaritySearch.getsatknnresult(ctx)
        q = Q[i]
        for (objID, d) in zip(idlist, distlist)
            objID == 0 && break
            d <= 0 && continue # needed to compute Q=X
            if SimilaritySearch.sat_should_push(IdView(res), dist, X, q, objID, d, near)
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

