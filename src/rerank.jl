# This file is part of SimilaritySearch.jl

export rerank!

"""
    rerank!(dist::PreMetric, db::AbstractDatabase, q, ids, dists) -> (ids, dists)

Re-scores and re-sorts, in place, an existing candidate result set `(ids, dists)` for query `q`
using `dist` as the exact (or otherwise more precise) distance function. This is typically used
to refine a result set obtained with a cheaper proxy distance or a lossy/approximate index.

# Arguments
- `dist`: the (typically exact) distance function used to re-score candidates
- `db`: the database that candidate identifiers in `ids` point into
- `q`: the query object
- `ids`: a vector of `UInt32` candidate identifiers; entries equal to `0` mark the end of valid candidates
- `dists`: a parallel vector of `Float32` distances to re-score

# Returns
`(ids, dists)`, sorted in ascending order by the recomputed distance (only over the valid, non-zero-id prefix).
"""
function rerank!(dist::PreMetric, db::AbstractDatabase, q,
                 ids::AbstractVector{UInt32}, dists::AbstractVector{Float32})
    m = 0
    for i in eachindex(ids)
        pid = ids[i]
        if pid == 0
            break
        else
            m = i
            o = db[pid]
            dists[i] = evaluate(dist, o, q)
        end
    end

    # Sort both arrays together by distance using a paired permutation sort
    p = sortperm(view(dists, 1:m))
    ids[1:m]   .= view(ids,   1:m)[p]
    dists[1:m] .= view(dists, 1:m)[p]
    ids, dists
end

"""
    rerank!(dist::PreMetric, db::AbstractDatabase, queries::AbstractDatabase,
            knns_ids::AbstractMatrix{UInt32}, knns_dists::AbstractMatrix{Float32}) -> (knns_ids, knns_dists)

Batch variant of [`rerank!`](@ref) that re-scores and re-sorts, in place and in parallel (one task per
query column), the candidate result set of every query in `queries`.

# Arguments
- `dist`: the (typically exact) distance function used to re-score candidates
- `db`: the database that candidate identifiers point into
- `queries`: the set of queries; its `i`-th element corresponds to the `i`-th column
- `knns_ids`: a `(k, n)` matrix of `UInt32` candidate identifiers (e.g., as produced by `searchbatch`)
- `knns_dists`: a `(k, n)` matrix of `Float32` distances, parallel to `knns_ids`

# Returns
`(knns_ids, knns_dists)`, with every column re-scored and sorted in ascending order by distance.

# Examples

```julia
using SimilaritySearch

exact_dist = Dist.L2()
proxy_dist = Dist.SqL2()
X = MatrixDatabase(rand(Float32, 8, 10^3))
Q = MatrixDatabase(rand(Float32, 8, 32))

E = ExhaustiveSearch(; dist=proxy_dist, db=X)
ctx = GenericContext()
ids, dists = searchbatch(E, ctx, Q, 8)

rerank!(exact_dist, X, Q, ids, dists)  # refines in place using the exact distance
```
"""
function rerank!(dist::PreMetric, db::AbstractDatabase, queries::AbstractDatabase,
                 knns_ids::AbstractMatrix{UInt32}, knns_dists::AbstractMatrix{Float32})
    m = length(queries)
    minbatch = getminbatch(m)
    @BATCHES minbatch for i in 1:m
        rerank!(dist, db, queries[i], view(knns_ids, :, i), view(knns_dists, :, i))
    end

    knns_ids, knns_dists
end

"""
    rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractKnnQueue) -> res

Re-scores and re-sorts, in place, an `AbstractKnnQueue` result object `res` for query `q` using `dist`.
"""
function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractKnnQueue)
    ids_v   = view(res.ids,   res.sp:res.ep)
    dists_v = view(res.dists, res.sp:res.ep)
    rerank!(dist, db, q, ids_v, dists_v)
    res
end
