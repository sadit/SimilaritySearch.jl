# This file is part of SimilaritySearch.jl

export hsp_queries

iterate_hsp_(h::Vector{T}) where {T<:Integer} = h
iterate_hsp_(h::AbstractKnnQueue) = IdView(h)
# the kept neighbors with their distance to the center, when the neighborhood carries them
iterate_hsp_pairs_(h::Vector{T}) where {T<:Integer} = (IdDist(UInt32(i), NaN32) for i in h)
iterate_hsp_pairs_(h::AbstractKnnQueue) = IdDistView(h)

"""
    hsp_should_push(hsp_neighborhood, dist, db, center, point_id, dist_center_point; factor=1f0, neardup=typemin(Float32)) -> Bool

Whether the candidate `point_id`, at `dist_center_point` from the center, survives the HSP
rule against the neighbors already kept: it is dropped when a kept neighbor is strictly
closer to it than the center is. `neardup` adds the tie rule for near duplicates of the
center: a kept neighbor within `neardup` of the center -- a twin, for which every candidate
is exactly as close as to the center -- drops candidates that are *at least* as close to it,
so a twin keeps nothing else and the duplicates of a point do not grow into a clique (see
[`Neighborhood`](@ref)'s `neardup`). The default never fires, so callers outside the graph
are unchanged.
"""
function hsp_should_push(hsp_neighborhood, dist::PreMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32; factor::Float32=1.0f0, neardup::Float32=typemin(Float32))
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
    # `point` and `hsp_obj` are both stored objects -- encoded, if the database encodes --
    # while the center never appears here: `dist_center_point` was evaluated by the caller
    # with the center as given (raw, in an asymmetric graph) against the stored point. An
    # estimator therefore evaluates two kinds of pairs: raw against stored to navigate and
    # to place the center, and stored against stored for this rule between candidates.
    @inbounds for h in iterate_hsp_pairs_(hsp_neighborhood)
        hsp_obj = db[h.id]
        dist_point_hsp = evaluate(dist, point, hsp_obj)
        # f * dist_point_hsp < dist_center_point && return false
        dist_point_hsp < dist_center_point && return false #  <= does not guarantee connectivity in all cases, but the insertion algorithm ensures that
        h.dist <= neardup && dist_point_hsp <= dist_center_point && return false   # the tie rule, for a twin of the center
    end

    true
end

"""
    hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase,
                knns_ids::AbstractMatrix{UInt32}, knns_dists::AbstractMatrix{Float32};
                scheduler::Symbol=get_batch_scheduler()) -> (ids, dists, hsp)

Computes the Half-Space Proximal (HSP) neighborhood of each query in `Q` by filtering its candidate
neighbors (given by `knns_ids`/`knns_dists`, e.g., as produced by `searchbatch`) so that only proximal,
non-redundant neighbors are kept.

# Arguments
- `dist`: the distance function used to evaluate candidates
- `X`: the database the candidate identifiers in `knns_ids` point into
- `Q`: the set of queries (its `i`-th element corresponds to the `i`-th column)
- `knns_ids`: a `(k, n)` matrix of `UInt32` identifiers (e.g., as produced by `searchbatch`)
- `knns_dists`: a `(k, n)` matrix of `Float32` distances, parallel to `knns_ids`

# Keyword Arguments
- `scheduler`: the [`@BATCHES`](@ref) scheduler used for the per-query HSP filtering
  (`:dynamic`, `:default`, `:static`, `:greedy`, or `:sequential` to disable threading entirely).
  Defaults to [`get_batch_scheduler`](@ref).

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
E = ExhaustiveSearch(dist, X)
ctx = GenericContext()

ids, dists = searchbatch(E, ctx, X, 32)
hsp_ids, hsp_dists, hsp = hsp_queries(dist, X, X, ids, dists)
length.(hsp)  # size of each query's HSP neighborhood
```
"""
function hsp_queries(dist, X::AbstractDatabase, Q::AbstractDatabase,
                     knns_ids::AbstractMatrix{UInt32}, knns_dists::AbstractMatrix{Float32};
                     scheduler::Symbol=get_batch_scheduler())
    k, n = size(knns_ids)
    @assert size(knns_dists) == (k, n)
    hsp_ids   = zeros(UInt32,  k, n)
    hsp_dists = fill(typemax(Float32), k, n)
    # KnnSorted iteration is in ascending order, not required here but consistent
    hsp = [knnqueue(KnnSorted, view(hsp_ids, :, i), view(hsp_dists, :, i)) for i in 1:n]
    minbatch = getminbatch(n)

    @BATCHES minbatch scheduler=scheduler for i in 1:n
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

function hsp_proximal_neighborhood_filter!(hsp::AbstractKnnQueue, dist::PreMetric, db, center, neighborhood; neardup::Float32=1.0f-4, neardupcaptureprob::Float32=0.5f0)
    push_item!(hsp, neighborhood[1])
    prob = 1.0f0 # ignore near duplicates with some prob
    for i in 2:length(neighborhood)
        p = neighborhood[i]
        if p.dist <= neardup
            if rand(Float32) < prob
                push_item!(hsp, p)
                prob *= neardupcaptureprob # workaround for very large number of duplicates
            end
        elseif hsp_should_push(hsp, dist, db, center, p.id, p.dist; neardup)
            push_item!(hsp, p)
        end
    end

    hsp
end

function hsp_distal_neighborhood_filter!(hsp::AbstractKnnQueue, dist::PreMetric, db, center, neighborhood)
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
