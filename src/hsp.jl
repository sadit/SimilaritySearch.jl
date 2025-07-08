# This file is part of SimilaritySearch.jl

export hsp_queries

iterate_hsp_(h::Vector{T}) where {T<:Integer} = h
iterate_hsp_(h::Vector{IdWeight}) = IdView(h)
iterate_hsp_(h::Knn) = IdView(h)
iterate_hsp_(h::XKnn) = IdView(h)

function hsp_should_push(hsp_neighborhood, dist::SemiMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32)
    @inbounds point = db[point_id]
    @inbounds for hsp_objID in iterate_hsp_(hsp_neighborhood)
        hsp_obj = db[hsp_objID]
        dist_point_hsp = evaluate(dist, point, hsp_obj)
        dist_point_hsp < dist_center_point && return false
    end

    true
end

#=
iterate_hsp_hyperbolic_(h::Knn) = eachiddist(h)
iterate_hsp_hyperbolic_(h::XKnn) = eachiddist(h)
iterate_hsp_hyperbolic_(h::Vector{IdWeight}) = eachiddist(h)
function iterate_hsp_hyperbolic_(hsp_neighborhood::Vector{T}, dist, center, db) where {T<:Integer} 
    (hsp_objID, evaluate(dist, center, db[hsp_objID]) for hsp_objID in hsp_neighborhood)
end
=#

#=
error("unsuported hfactor $hfactor")
function #hyperbolic_hsp_should_push(hsp_neighborhood, dist::SemiMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32, hfactor::Float32)
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

=#


"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::Matrix; <kwargs>)
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; <kwargs>)
    hsp_queries(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer; <kwargs>)


Computes the half-space partition of the queries `Q` (possibly given as a `knns` of `IdWeight` elements)


## Optional keyword arguments
- `ctx::SearchGraphContext` search context (caches)
- `minbatch::Int`: `Polyester.@batch` parameter controlling how the multithreading is executed
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::AbstractMatrix; ctx = SearchGraphContext(), minbatch::Int=0)
    n = length(Q)
    hsp = xknnset(size(knns)...)

    minbatch = getminbatch(minbatch, n)
    @batch minbatch=minbatch per=thread for i in 1:n
        plist = @view knns[:, i]
        q = Q[i]
        for p in plist
            p.id == 0 && break
            if hsp_should_push(hsp.knns[i], dist, X, q, p.id, p.weight)
                push_item!(hsp.knns[i], p)
            end
        end

    end

    hsp
end

function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, k::Integer; minbatch=0)
    idx = ExhaustiveSearch(; dist, db=X)
    ctx = getcontext(idx)
    knns = searchbatch(idx, ctx, Q, k)
    hsp_queries(dist, X, Q, knns; ctx, minbatch)
end

function hsp_queries(idx::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer; minbatch=0)
    knns = searchbatch(idx, ctx, Q, k)
    hsp_queries(distance(idx), database(idx), Q, knns; ctx, minbatch)
end

function hsp_proximal_neighborhood_filter!(hsp, dist::SemiMetric, db, item, neighborhood; hfactor::Float32=0f0, nndist::Float32=1f-4, nncaptureprob::Float32=0.5f0)
    push_item!(hsp, neighborhood[1])
    prob = 1f0
    #hfactor = 0.9f0
    #hfactor_gain = 1.05f0
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.weight <= nndist
            if rand(Float32) < prob
                push_item!(hsp, p)
                prob *= nncaptureprob # workaround for very large number of duplicates
            end
        elseif hfactor == 0f0
            hsp_should_push(hsp, dist, db, item, p.id, p.weight) && push_item!(hsp, p)
        else
            error("unsuported hfactor $hfactor")
            #hyperbolic_hsp_should_push(hsp, dist, db, item, p.id, p.weight, hfactor) && push_item!(hsp, p.id, p.weight)
        end
    end
end

function hsp_distal_neighborhood_filter!(hsp, dist::SemiMetric, db, item, neighborhood; hfactor::Float32=0f0, nndist::Float32=1f-4)
    push_item!(hsp, last(neighborhood))

    @inbounds for i in length(neighborhood)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        p = neighborhood[i]
        if p.weight <= nndist
            push_item!(hsp, p)
        elseif hfactor == 0f0
            hsp_should_push(hsp, dist, db, item, p.id, p.weight) && push_item!(hsp, p)
        else
            error("unsuported hfactor $hfactor")
            # hyperbolic_hsp_should_push(hsp, dist, db, item, p.id, p.weight, hfactor) && push_item!(hsp, p.id, p.weight)
        end
    end

end
