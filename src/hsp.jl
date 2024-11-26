# This file is part of SimilaritySearch.jl

export hsp_queries

iterate_hsp_(hsp_neighborhood::Vector{T}) where {T<:Integer} = hsp_neighborhood
iterate_hsp_(hsp_neighborhood::Vector{IdWeight}) = eachid(hsp_neighborhood)
iterate_hsp_(hsp_neighborhood::KnnResult) = eachid(hsp_neighborhood)

function hsp_should_push(hsp_neighborhood, dist::SemiMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32)
    @inbounds point = db[point_id]
    @inbounds for hsp_objID in iterate_hsp_(hsp_neighborhood)
        hsp_obj = db[hsp_objID]
        dist_point_hsp = evaluate(dist, point, hsp_obj)
        dist_point_hsp < dist_center_point && return false
    end

    true
end


iterate_hsp_hyperbolic_(hsp_neighborhood::KnnResult) = eachiddist(hsp_neighborhood)
iterate_hsp_hyperbolic_(hsp_neighborhood::Vector{IdWeight}) = eachiddist(hsp_neighborhood)
function iterate_hsp_hyperbolic_(hsp_neighborhood::Vector{T}, dist, center, db) where {T<:Integer} 
    (hsp_objID, evaluate(dist, center, db[hsp_objID]) for hsp_objID in hsp_neighborhood)
end


function hyperbolic_hsp_should_push(hsp_neighborhood, dist::SemiMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32, hfactor::Float32)
    @inbounds point = db[point_id]
    @inbounds for (hsp_objID, dist_center_hsp) in iterate_hsp_hyperbolic_(hsp_neighborhood, dist, center, db)
        dist_point_hsp = evaluate(dist, point, db[hsp_objID])
        abs(dist_point_hsp - dist_center_point) <= hfactor * dist_center_hsp && return false
    end
    #=@inbounds for (hsp_objID, dist_center_hsp) in eachiddist(hsp_neighborhood)
        hsp_obj = db[hsp_objID]
        dist_point_hsp = evaluate(dist, point, hsp_obj)
        dist_point_hsp / dist_center_hsp <= hfactor && return false
        # abs(dist_point_hsp - dist_center_point) <= hfactor * dist_center_hsp && return false
    end=#

    true 
end




"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns, dists; <kwargs>)
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; <kwargs>)
    hsp_queries(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer; <kwargs>)


Computes the half-space partition of the queries `Q` (possibly given as `knns`, `dists`)


## Optional keyword arguments
- `ctx::SearchGraphContext` search context (caches)
- `minbatch::Int`: `Polyester.@batch` parameter controlling how the multithreading is executed
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::AbstractMatrix, dists::AbstractMatrix; ctx = SearchGraphContext(), minbatch::Int=0)
    n = length(Q)
    hsp = Vector{KnnResult}(undef, n)
    minbatch = getminbatch(minbatch, n)
    @batch minbatch=minbatch per=thread for i in 1:n
        idlist = @view knns[:, i]
        distlist = @view dists[:, i]
        res = getknnresult(length(idlist), ctx)
        q = Q[i]
        for (objID, d) in zip(idlist, distlist)
            objID == 0 && break
            if hsp_should_push(res, dist, X, q, convert(UInt32, objID), convert(Float32, d))
                push_item!(res, objID, d)
            end
        end

        hsp[i] = KnnResult(copy(res.items), length(res))
    end

    hsp
end

function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; minbatch=0)
    idx = ExhaustiveSearch(; dist, db=X)
    ctx = getcontext(idx)
    knns, dists = searchbatch(idx, ctx, Q, k)
    hsp_queries(dist, X, Q, knns, dists; ctx, minbatch)
end

function hsp_queries(idx::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer; minbatch=0)
    knns, dists = searchbatch(idx, ctx, Q, k)
    hsp_queries(distance(idx), database(idx), Q, knns, dists; ctx, minbatch)
end

function hsp_proximal_neighborhood_filter!(hsp_neighborhood::KnnResult, dist::SemiMetric, db, item, neighborhood::KnnResult; hfactor::Float32=0f0, nndist::Float32=1f-4, nncaptureprob::Float32=0.5f0)
    push_item!(hsp_neighborhood, argmin(neighborhood), minimum(neighborhood)) 
    prob = 1f0
    #hfactor = 0.9f0
    #hfactor_gain = 1.05f0
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.weight <= nndist
            if rand(Float32) < prob
                push_item!(hsp_neighborhood, p)
                prob *= nncaptureprob # workaround for very large number of duplicates
            end
        elseif hfactor == 0f0
            hsp_should_push(hsp_neighborhood, dist, db, item, p.id, p.weight) && push_item!(hsp_neighborhood, p.id, p.weight)
        else
            hyperbolic_hsp_should_push(hsp_neighborhood, dist, db, item, p.id, p.weight, hfactor) && push_item!(hsp_neighborhood, p.id, p.weight)
        end
    end
end

function hsp_distal_neighborhood_filter!(hsp_neighborhood::KnnResult, dist::SemiMetric, db, item, neighborhood::KnnResult; hfactor::Float32=0f0, nndist::Float32=1f-4)
    push_item!(hsp_neighborhood, argmax(neighborhood), maximum(neighborhood))

    @inbounds for i in length(neighborhood)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        p = neighborhood[i]
        if p.weight <= nndist
            push_item!(hsp_neighborhood, p)
        elseif hfactor == 0f0
            hsp_should_push(hsp_neighborhood, dist, db, item, p.id, p.weight) && push_item!(hsp_neighborhood, p.id, p.weight)
        else
            hyperbolic_hsp_should_push(hsp_neighborhood, dist, db, item, p.id, p.weight, hfactor) && push_item!(hsp_neighborhood, p.id, p.weight)
        end
    end

end
