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

Computes the half-space partition of the queries `Q` (possibly given as a `knns` of `IdWeight` elements)

## Optional keyword arguments
- `minbatch::Int`: `Polyester.@batch` parameter controlling how the multithreading is executed
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::AbstractMatrix; minbatch::Int=0)
    n = length(Q)
    matrix = zeros(eltype(knns), size(knns)...)
    hsp = [knndefault(c) for c in eachcol(matrix)]
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        plist = @view knns[:, i]
        q = Q[i]
        for p in plist
            p.id == 0 && break
            if hsp_should_push(hsp[i], dist, X, q, p.id, p.weight)
                hsp[i], _ = push_item!(hsp[i], p)
            end
        end

    end

    matrix, hsp
end

function hsp_proximal_neighborhood_filter!(hsp, neighborhood; nndist::Float32=1f-4, nncaptureprob::Float32=0.5f0)
    hsp, _ = push_item!(hsp, neighborhood[1])
    prob = 1f0
    #hfactor = 0.9f0
    #hfactor_gain = 1.05f0
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.weight <= nndist
            if rand(Float32) < prob
                hsp, _ = push_item!(hsp, p)
                prob *= nncaptureprob # workaround for very large number of duplicates
            end
        end
    end

    hsp
end

function hsp_distal_neighborhood_filter!(hsp, neighborhood; nndist::Float32=1f-4)
    hsp, _ = push_item!(hsp, last(neighborhood))

    @inbounds for i in length(neighborhood)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        p = neighborhood[i]
        if p.weight <= nndist
            hsp, _ = push_item!(hsp, p)
        end
    end

    hsp
end
