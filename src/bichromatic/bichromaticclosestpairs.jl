# This file is a part of SimilaritySearch.jl

"""
    bichromatic_kclosestpairs(idxA::AbstractSearchIndex, ctx::AbstractContext, B::AbstractDatabase; k::Int=1, min_k::Int=max(k, 8), samedata::Bool=database(idxA) === B) -> Vector{Tuple{Int32,Int32,Float32}}

Finds the `k` closest pairs `(a, b)` between dataset `A` (already indexed as `idxA`) and dataset `B`
(queried directly, with no index of its own) -- the same idea as [`bichromatic_closestpair`](@ref)
(which is exactly the `k == 1` case), generalized from "the single globally closest pair" to "the `k`
globally closest pairs". If `idxA` is an approximate index then the resulting pairs may also be an
approximation of the true `k` closest pairs.

Self-match exclusion works exactly as in [`bichromatic_closestpair`](@ref): controlled by `samedata`,
defaulting to `database(idxA) === B`.

Each `b ∈ B` is searched for its `min_k` nearest candidates in `idxA` (not just its single nearest, as
[`bichromatic_closestpair`](@ref) does), since a single `b` may contribute more than one of the `k`
globally closest pairs. `min_k` must be `>= k` for the result to be exact on an exact index (the
default `max(k, 8)` guarantees this); a smaller `min_k` would silently cap how many pairs a single `b`
can contribute. Every batch keeps its own bounded (`<= k`) ascending buffer of candidate pairs, which
are merged into the final top `k` once every batch finishes -- the same per-batch-then-merge structure
[`bichromatic_closestpair`](@ref) uses for a single best pair.

# Arguments
- `idxA`: the search structure indexing dataset `A`
- `ctx`: the search context used by `idxA` (caches, hyperparameters, scheduler, etc.)
- `B`: the dataset queried against `idxA`, with no index of its own

# Keyword Arguments
- `k`: how many globally closest pairs to return
- `min_k`: candidate buffer size per query into `idxA`; must be `>= k` for exactness (see above)
- `samedata`: whether `idxA` indexes `B` itself, i.e. whether self-matches must be excluded

# Returns
Up to `k` tuples `(i, j, dist)` (identifier `i` of `idxA`, identifier `j` of `B`, and their distance),
sorted ascending by distance. Fewer than `k` tuples are returned if there aren't that many eligible
pairs (e.g. a very small dataset with `samedata == true`).

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
A = MatrixDatabase(rand(Float32, 2, 10^3))
B = MatrixDatabase(rand(Float32, 2, 10^3))
GA = SearchGraph(dist, A)
ctx = SearchGraphContext()
index!(GA, ctx)

pairs = bichromatic_kclosestpairs(GA, ctx, B; k=10)
```
"""
function bichromatic_kclosestpairs(idxA::T, ctx::AbstractContext, B::AbstractDatabase;
        k::Int=1,
        min_k::Int=max(k, 8),
        samedata::Bool=database(idxA) === B
    )::Vector{Tuple{Int32,Int32,Float32}} where {T<:AbstractSearchIndex}
    n = length(B)
    minbatch = getminbatch(ctx, n)
    local best

    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGIN
        # one column/slot per batch -- @batchid()-indexed, so race-free regardless of scheduler
        knns_ids = zeros(UInt32, min_k, @nbatches())
        knns_dists = zeros(Float32, min_k, @nbatches())
        BEST = Vector{Vector{Tuple{Int32,Int32,Float32}}}(undef, @nbatches())
    @BEGINBATCH
        batchctx = @set ctx.batchid = @batchid()
        r = knnqueue(KnnSorted, view(knns_ids, :, @batchid()), view(knns_dists, :, @batchid())) # requires KnnSorted to support pop_min!
        topk = Tuple{Int32,Int32,Float32}[]  # this batch's own bounded (<= k) ascending top-k buffer
        sizehint!(topk, k)
    @LOOP for objID in 1:n
        res = bichromatic_search!(idxA, batchctx, B, objID, reuse!(r), samedata)
        insert_topk_candidates!(topk, k, res, objID)
    end
    @ENDBATCH
        BEST[@batchid()] = topk
    @END
        best = merge_topk(BEST, k)
    end

    best
end

"""
    insert_topk_candidates!(topk, k, res, objID) -> topk

Inserts every candidate in `res` (a `bichromatic_search!` result, sorted ascending by distance) as an
`(id_in_A, objID, dist)` triple into `topk`, a bounded (`<= k`) ascending buffer. Stops early once a
candidate is no worse than `topk`'s current worst entry and `topk` is already full, since `res` being
sorted means every remaining candidate is at least as bad.
"""
function insert_topk_candidates!(topk::Vector{Tuple{Int32,Int32,Float32}}, k::Int, res, objID::Integer)
    for p in IdDistView(res)
        length(topk) >= k && p.dist >= last(topk[end]) && break
        insert_topk!(topk, k, (Int32(p.id), Int32(objID), p.dist))
    end

    topk
end

"""
    insert_topk!(topk, k, cand) -> topk

Inserts `cand` into the bounded (`<= k`) ascending-by-distance buffer `topk`, evicting the current
worst entry if `topk` was already at capacity `k`. `cand` is a `(i, j, dist)` triple, ordered by `dist`
(its `last` component).
"""
function insert_topk!(topk::Vector{Tuple{Int32,Int32,Float32}}, k::Int, cand::Tuple{Int32,Int32,Float32})
    if length(topk) < k
        insert!(topk, searchsortedfirst(topk, cand; by=last), cand)
    elseif last(cand) < last(topk[end])
        insert!(topk, searchsortedfirst(topk, cand; by=last), cand)
        pop!(topk)
    end

    topk
end

"""
    merge_topk(BEST, k) -> Vector{Tuple{Int32,Int32,Float32}}

Merges the per-batch top-`k` buffers `BEST` (each already `<= k` and ascending) into the single global
top `k`, ascending.
"""
function merge_topk(BEST::Vector{Vector{Tuple{Int32,Int32,Float32}}}, k::Int)
    merged = reduce(vcat, BEST)
    sort!(merged; by=last)
    merged[1:min(k, length(merged))]
end
