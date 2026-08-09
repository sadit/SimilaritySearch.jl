# This file is part of SimilaritySearch.jl

export hsp_queries

iterate_hsp_(h::Vector{T}) where {T<:Integer} = h
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
        dist_point_hsp < dist_center_point && return false #  <= does not guarantee connectivity in all cases, but the insertion algorithm ensures that
    end

    true
end

"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase,
                knns_ids::AbstractMatrix{UInt32}, knns_dists::AbstractMatrix{Float32}) -> (ids, dists, hsp)

Computes the Half-Space Proximal (HSP) neighborhood of each query in `Q` by filtering its candidate
neighbors (given by `knns_ids`/`knns_dists`, e.g., as produced by `searchbatch`) so that only proximal,
non-redundant neighbors are kept.

# Arguments
- `dist`: the distance function used to evaluate candidates
- `X`: the database the candidate identifiers in `knns_ids` point into
- `Q`: the set of queries (its `i`-th element corresponds to the `i`-th column)
- `knns_ids`: a `(k, n)` matrix of `UInt32` identifiers (e.g., as produced by `searchbatch`)
- `knns_dists`: a `(k, n)` matrix of `Float32` distances, parallel to `knns_ids`

# Returns
A tuple `(hsp_ids, hsp_dists, hsp)` where:
- `hsp_ids`: a `(k, n)` matrix of `UInt32` identifiers backing the `hsp` result objects
- `hsp_dists`: a `(k, n)` matrix of `Float32` distances backing the `hsp` result objects
- `hsp`: a vector of `KnnSorted` objects, one per query, containing its HSP-filtered neighborhood

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 10^3))
E = ExhaustiveSearch(; dist, db=X)
ctx = GenericContext()

ids, dists = searchbatch(E, ctx, X, 32)
hsp_ids, hsp_dists, hsp = hsp_queries(dist, X, X, ids, dists)
length.(hsp)  # size of each query's HSP neighborhood
```
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase,
                     knns_ids::AbstractMatrix{UInt32}, knns_dists::AbstractMatrix{Float32})
    k, n = size(knns_ids)
    @assert size(knns_dists) == (k, n)
    hsp_ids   = zeros(UInt32,  k, n)
    hsp_dists = fill(typemax(Float32), k, n)
    # KnnSorted iteration is in ascending order, not required here but consistent
    hsp = [knnqueue(KnnSorted, view(hsp_ids, :, i), view(hsp_dists, :, i)) for i in 1:n]
    minbatch = getminbatch(n)

    @BATCHES minbatch for i in 1:n
        q = Q[i]
        for j in 1:k
            pid  = knns_ids[j, i]
            pid == 0 && break
            pdist = knns_dists[j, i]
            if hsp_should_push(hsp[i], dist, X, q, pid, pdist)
                push_item!(hsp[i], pid, pdist)
            end
        end
    end

    hsp_ids, hsp_dists, hsp
end

function hsp_proximal_neighborhood_filter!(hsp::AbstractKnn, dist::PreMetric, db, center, neighborhood; neardup::Float32=1.0f-4, neardupcaptureprob::Float32=0.5f0)
    push_item!(hsp, neighborhood[1])
    prob = 1.0f0 # ignore near duplicates with some prob
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.dist <= neardup
            if rand(Float32) < prob
                push_item!(hsp, p)
                prob *= neardupcaptureprob # workaround for very large number of duplicates
            end
        elseif hsp_should_push(hsp, dist, db, center, p.id, p.dist)
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
        if hsp_should_push(hsp, dist, db, center, p.id, p.dist)
            push_item!(hsp, p)
        end
    end

    hsp
end
