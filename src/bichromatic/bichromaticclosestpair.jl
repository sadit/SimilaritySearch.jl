# This file is a part of SimilaritySearch.jl

"""
    bichromatic_closestpair(idxA::AbstractSearchIndex, ctx::AbstractContext, B::AbstractDatabase; min_k::Int=8, samedata::Bool=database(idxA) === B) -> (i, j, dist)

Finds the closest pair `(a, b)` with `a` an identifier of `idxA` and `b` an identifier of `B`, i.e.,
the closest pair between dataset `A` (already indexed as `idxA`) and dataset `B` (queried directly,
with no index of its own). If `idxA` is an approximate index then the resulting pair may also be an
approximation of the true closest pair.

If `database(idxA) === B` (i.e. `idxA` indexes `B` itself, as [`closestpair`](@ref) uses), self-matches
(an element paired with itself) are excluded from the result; otherwise `A` and `B` are assumed to be
disjoint datasets and every candidate pair is eligible. `samedata` defaults to this check but can be
overridden explicitly.

This function always iterates over `B`'s elements querying into `idxA`; it does not (yet) pick the
smaller side to minimize the number of queries.

Always uses the `@BATCHES`-driven implementation (there is no separate sequential path) -- on a single
thread `@BATCHES` itself collapses to a single serial batch, so there is no parallelism overhead to avoid.

# Arguments
- `idxA`: the search structure indexing dataset `A`
- `ctx`: the search context used by `idxA` (caches, hyperparameters, scheduler, etc.)
- `B`: the dataset queried against `idxA`, with no index of its own

# Keyword Arguments
- `min_k`: instead of looking for `k=1` some approximate methods can take advantage of a larger `k`
  (also needed for stability: must be `>= 2` when `samedata == true`, since one slot is spent on the
  excluded self-match)
- `samedata`: whether `idxA` indexes `B` itself, i.e. whether self-matches must be excluded

# Returns
A tuple `(i, j, dist)` with the identifier `i` of `idxA`, the identifier `j` of `B`, and their
distance `dist`.

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
A = MatrixDatabase(rand(Float32, 2, 10^3))
B = MatrixDatabase(rand(Float32, 2, 10^3))
GA = SearchGraph(dist, A)
ctx = SearchGraphContext()
index!(GA, ctx)

i, j, d = bichromatic_closestpair(GA, ctx, B)
```
"""
function bichromatic_closestpair(idxA::T, ctx::AbstractContext, B::AbstractDatabase;
        min_k::Int=8,
        samedata::Bool=database(idxA) === B
    )::Tuple{Int32,Int32,Float32} where {T<:AbstractSearchIndex}
    n = length(B)
    minbatch = getminbatch(ctx, n)
    local best

    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGIN
        # one column/slot per batch -- @batchid()-indexed, so race-free regardless of scheduler
        knns_ids = zeros(UInt32, min_k, @nbatches())
        knns_dists = zeros(Float32, min_k, @nbatches())
        BEST = Vector{Tuple{Int32,Int32,Float32}}(undef, @nbatches())
    @BEGINBATCH
        batchctx = @set ctx.batchid = @batchid()
        r = knnqueue(KnnSorted, view(knns_ids, :, @batchid()), view(knns_dists, :, @batchid())) # requires KnnSorted to support pop_min!
        b = (zero(Int32), zero(Int32), typemax(Float32))
    @LOOP for objID in 1:n
        p = bichromatic_search_hint(idxA, batchctx, B, objID, reuse!(r), samedata)

        if p.dist < last(b)
            b = (p.id, Int32(objID), p.dist)
        end
    end
    @ENDBATCH
        BEST[@batchid()] = b
    @END
        _, i = findmin(last, BEST)
        best = BEST[i]
    end

    best
end

"""
    bichromatic_search!(idxA, ctx, B, i, res, samedata::Bool) -> res

Queries `idxA` (via `ctx`) with the `i`-th element of `B`, reusing `res` as the output buffer, and
returns `res` itself (sorted ascending by distance). `samedata` controls whether a match with the
very same identifier `i` is dropped -- needed when `idxA` indexes `B` itself (see
[`bichromatic_closestpair`](@ref)).
"""
function bichromatic_search!(idxA::AbstractSearchIndex, ctx::AbstractContext, B::AbstractDatabase, i::Integer, res, samedata::Bool)
    res = search(idxA, ctx, B[i], res)
    if samedata && argmin(res) == i
        pop_min!(res)
    end

    res
end

function bichromatic_search!(idxA::SearchGraph, ctx::SearchGraphContext, B::AbstractDatabase, i::Integer, res, samedata::Bool)
    vstate = getvstate(length(idxA), ctx)
    if samedata
        # i is a valid vertex of idxA's own graph here (idxA indexes B) -- mark it visited so
        # the beam doesn't waste work expanding into the query's own node, and seed the beam
        # from one of its neighbors instead of the index's default hints.
        visit!(vstate, convert(UInt64, i))
        hints = rand(neighbors(idxA.adj, i))
        search(idxA.algo[], idxA, ctx, B[i], res, hints, vstate)
    else
        search(idxA.algo[], idxA, ctx, B[i], res, idxA.hints, vstate)
    end

    if samedata && argmin(res) == i
        pop_min!(res)
    end

    res
end

"""
    bichromatic_search_hint(idxA, ctx, B, i, res, samedata::Bool)

The nearest match of [`bichromatic_search!`](@ref), i.e. `nearest(bichromatic_search!(...))`.
"""
bichromatic_search_hint(idxA::AbstractSearchIndex, ctx::AbstractContext, B::AbstractDatabase, i::Integer, res, samedata::Bool) =
    nearest(bichromatic_search!(idxA, ctx, B, i, res, samedata))

