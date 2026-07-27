# This file is part of SimilaritySearch.jl

export rerank!

"""
    rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractVector{IdDist}) -> res

Re-scores and re-sorts, in place, an existing candidate result set `res` for query `q` using `dist` as the
exact (or otherwise more precise) distance function. This is typically used to refine a result set that was
obtained with a cheaper proxy distance or a lossy/approximate index, since it corrects the reported distances
and their relative order.

# Arguments
- `dist`: the (typically exact) distance function used to re-score candidates
- `db`: the database that candidate identifiers in `res` point into
- `q`: the query object
- `res`: a vector of `IdDist` candidates for `q` (e.g., a column of a `knns` matrix); ids equal to `0`
  are treated as unused slots and mark the end of the valid candidates

# Returns
`res`, sorted in ascending order by the recomputed distance (only over its valid, non-zero-id prefix).
"""
function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractVector{IdDist})
    m = 0
    for i in eachindex(res)
        p = res[i]
        if p.id == 0
            break
        else
            m = i
            o = db[p.id]
            d = evaluate(dist, o, q)
            res[i] = IdDist(p.id, d)
        end
    end

    sort!(view(res, 1:m), by=x -> x.dist)
    res
end

"""
    rerank!(dist::PreMetric, db::AbstractDatabase, queries::AbstractDatabase, knns::AbstractMatrix{IdDist}) -> knns

Batch variant of [`rerank!`](@ref) that re-scores and re-sorts, in place and in parallel (one task per query
column), the candidate result set of every query in `queries`. This is the main entry point of this file and
is typically used after a batch approximate search to refine its results with an exact distance function.

# Arguments
- `dist`: the (typically exact) distance function used to re-score candidates
- `db`: the database that candidate identifiers in `knns` point into
- `queries`: the set of queries; its `i`-th element corresponds to the `i`-th column of `knns`
- `knns`: a `(k, n)` matrix of `IdDist` candidates, one column per query (e.g., as produced by `searchbatch!`)

# Returns
`knns`, with every column re-scored and sorted in ascending order by the recomputed distance.

# Examples

```julia
using SimilaritySearch

exact_dist = Dist.L2()
proxy_dist = Dist.SqL2()
X = MatrixDatabase(rand(Float32, 8, 10^3))
Q = MatrixDatabase(rand(Float32, 8, 32))

E = ExhaustiveSearch(; dist=proxy_dist, db=X)
ctx = getcontext(E)
knns = searchbatch(E, ctx, Q, 8)

rerank!(exact_dist, X, Q, knns)  # refines knns in place using the exact distance
```
"""
function rerank!(dist::PreMetric, db::AbstractDatabase, queries::AbstractDatabase, knns::AbstractMatrix{IdDist})
    m = length(queries)
    minbatch = getminbatch(m, Threads.nthreads(), 0)
    @batch per=thread minbatch=minbatch for i in 1:m
        res = view(knns, :, i)
        rerank!(dist, db, queries[i], res)
    end

    knns
end

"""
    rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractKnn) -> res

Re-scores and re-sorts, in place, an `AbstractKnn` result object `res` for query `q` using `dist`, by
delegating to the `AbstractVector{IdDist}` method over `viewitems(res)`. See [`rerank!`](@ref) for details.
"""
function rerank!(dist::PreMetric, db::AbstractDatabase, q, res::AbstractKnn)
    rerank!(dist, db, q, viewitems(res))
end

