# This file is part of SimilaritySearch.jl
#

export hsp_queries

function hsp_should_push(hsp_neighborhood::Vector{T}, dfun::SemiMetric, db::AbstractDatabase, center, point_id::UInt32, dist_between_point_and_center::Float32, hfactor::Float32) where {T<:Integer}
    @inbounds tested_point = db[point_id]
    if hfactor <= eps(Float32)
        @inbounds for hsp_objID in hsp_neighborhood
            hsp_obj = db[hsp_objID]
            d = evaluate(dfun, tested_point, hsp_obj)
            d < dist_between_point_and_center && return false
        end
    else
        @inbounds for hsp_objID in hsp_neighborhood
            hsp_obj = db[hsp_objID]
            d = evaluate(dfun, tested_point, hsp_obj)
            d - dist_between_point_and_center < hfactor * evaluate(dfun, center, hsp_obj) && return false
        end
    end

    true 
end

function hsp_should_push(hsp_neighborhood::Union{KnnResult,Vector{IdWeight}}, dfun::SemiMetric, db::AbstractDatabase, center, point_id::UInt32, dist_between_point_and_center::Float32, hfactor::Float32)
    @inbounds tested_point = db[point_id]
    if hfactor <= eps(Float32)
        @inbounds for hsp_objID in eachid(hsp_neighborhood)
            hsp_obj = db[hsp_objID]
            d = evaluate(dfun, tested_point, hsp_obj)
            d < dist_between_point_and_center && return false
        end
    else
        @inbounds for (hsp_objID, dist_between_hsp_obj_and_center) in eachiddist(hsp_neighborhood)
            hsp_obj = db[hsp_objID]
            d = evaluate(dfun, tested_point, hsp_obj)
            d - dist_between_point_and_center < hfactor * dist_between_hsp_obj_and_center && return false
        end
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
- `hfactor::Float32` hyperbolic parameter, hfactor=0 means for hyperplane partitions
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::AbstractMatrix, dists::AbstractMatrix; ctx = SearchGraphContext(), hfactor::Float32=0f0)
    n = length(Q)
    hsp = Vector{KnnResult}(undef, n)
    Threads.@threads :static for i = 1:n
        idlist = @view knns[:, i]
        distlist = @view dists[:, i]
        res = getknnresult(length(idlist), ctx)
        q = Q[i]
        for (objID, d) in zip(idlist, distlist)
            objID == 0 && break
            if hsp_should_push(res, dist, X, q, convert(UInt32, objID), convert(Float32, d), hfactor)
                push_item!(res, objID, d)
            end
        end

        hsp[i] = KnnResult(copy(res.items), length(res))
    end

    hsp
end

function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext(), hfactor::Float32=0f0)
    idx = ExhaustiveSearch(; dist, db=X)
    knns, dists = searchbatch(idx, Q, k)
    hsp_queries(dist, X, Q, knns, dists; ctx, hfactor)
end

function hsp_queries(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer; ctx=SearchGraphContext(), hfactor::Float32=0f0)
    knns, dists = searchbatch(idx, ctx, Q, k)
    hsp_queries(distance(idx), database(idx), Q, knns, dists; ctx, hfactor)
end

