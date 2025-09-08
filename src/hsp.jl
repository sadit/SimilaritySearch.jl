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



"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::Matrix; <kwargs>)

Computes the half-space partition of the queries `Q` (possibly given as a `knns` of `IdWeight` elements)

## Optional keyword arguments
- `minbatch::Int`: `Polyester.@batch` parameter controlling how the multithreading is executed
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::AbstractMatrix; minbatch::Int=0)
    n = length(Q)
    matrix = zeros(IdWeight, size(knns)...)
    hsp = [xknn(c) for c in eachcol(matrix)]
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        plist = @view knns[:, i]
        q = Q[i]
        for p in plist
            p.id == 0 && break
            if hsp_should_push(hsp[i], dist, X, q, p.id, p.weight)
                push_item!(hsp[i], p)
            end
        end

    end

    matrix, hsp
end

function hsp_proximal_neighborhood_filter!(hsp, dist::SemiMetric, db, center, neighborhood; nndist::Float32=1f-4, nncaptureprob::Float32=0.5f0)
    push_item!(hsp, neighborhood[1])
    prob = 1f0
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.weight <= nndist
            if rand(Float32) < prob
                push_item!(hsp, p)
                prob *= nncaptureprob # workaround for very large number of duplicates
            end
        elseif hsp_should_push(hsp, dist, db, center, p.id, p.weight)
            push_item!(hsp, p)
        end
    end

    hsp
end

function hsp_distal_neighborhood_filter!(hsp, dist::SemiMetric, db, center, neighborhood; nndist::Float32=1f-4)
    push_item!(hsp, last(neighborhood))

    @inbounds for i in length(neighborhood)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        p = neighborhood[i]
        if p.weight <= nndist
            push_item!(hsp, p)
        elseif hsp_should_push(hsp, dist, db, center, p.id, p.weight)
            push_item!(hsp, p)
        end
    end

    hsp
end
