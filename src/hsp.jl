# This file is part of SimilaritySearch.jl

export hsp_queries

iterate_hsp_(h::Vector{T}) where {T<:Integer} = h
iterate_hsp_(h::Vector{IdWeight}) = IdView(h)
iterate_hsp_(h::AbstractKnn) = IdView(h)

function hsp_should_push(hsp_neighborhood, dist::PreMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32; factor::Float32=1.0f0)
    @inbounds point = db[point_id]
    #=if factor == 1.0f0
        @inbounds for hsp_objID in iterate_hsp_(hsp_neighborhood)
            hsp_obj = db[hsp_objID]
            dist_point_hsp = evaluate(dist, point, hsp_obj)
            dist_point_hsp < dist_center_point && return false
        end
    else
        f = Float32(factor)
        @inbounds for hsp_objID in iterate_hsp_(hsp_neighborhood)
            hsp_obj = db[hsp_objID]
            dist_point_hsp = evaluate(dist, point, hsp_obj)
            f * dist_point_hsp < dist_center_point && return false
            f = (f + 1.0f0) * 0.5f0
        end
    end=#
    @inbounds for hsp_objID in iterate_hsp_(hsp_neighborhood)
        hsp_obj = db[hsp_objID]
        dist_point_hsp = evaluate(dist, point, hsp_obj)
        # f * dist_point_hsp < dist_center_point && return false
        dist_point_hsp < dist_center_point && return false
    end

    true
end



"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::Matrix; <kwargs>)

Computes the half-space partition of the queries `Q` (possibly given as a `knns` of `IdWeight` elements)

"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase, knns::AbstractMatrix)
    n = length(Q)
    matrix = zeros(IdWeight, size(knns)...)
    # KnnSorted iteration is made in ascending order but it is not required here, so it can be changed if we expect a very high hsp
    hsp = [knnqueue(KnnSorted, c) for c in eachcol(matrix)]
    minbatch = getminbatch(n, Threads.nthreads(), 0)

    Threads.@threads :static for j in 1:minbatch:n
        for i in j:min(n, j + minbatch - 1)
            plist = @view knns[:, i]
            q = Q[i]
            for p in plist
                p.id == 0 && break
                if hsp_should_push(hsp[i], dist, X, q, p.id, p.weight)
                    push_item!(hsp[i], p)
                end
            end
        end
    end

    matrix, hsp
end

function hsp_proximal_neighborhood_filter!(hsp::AbstractKnn, dist::PreMetric, db, center, neighborhood; neardup::Float32=1.0f-4, neardupcaptureprob::Float32=0.5f0)
    push_item!(hsp, neighborhood[1])
    prob = 1.0f0 # ignore near duplicates with some prob
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.weight <= neardup
            if rand(Float32) < prob
                push_item!(hsp, p)
                prob *= neardupcaptureprob # workaround for very large number of duplicates
            end
        elseif hsp_should_push(hsp, dist, db, center, p.id, p.weight)
            push_item!(hsp, p)
        end
    end

    hsp
end

function hsp_distal_neighborhood_filter!(hsp::AbstractKnn, dist::PreMetric, db, center, neighborhood)
    push_item!(hsp, last(neighborhood))

    # prob = 1f0
    @inbounds for i in length(neighborhood)-1:-1:1  # DistSat produces larger neighborhoods
        p = neighborhood[i]
        if hsp_should_push(hsp, dist, db, center, p.id, p.weight)
            push_item!(hsp, p)
        end
    end

    hsp
end
