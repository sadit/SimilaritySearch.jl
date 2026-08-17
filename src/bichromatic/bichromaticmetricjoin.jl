# This file is a part of SimilaritySearch.jl

"""
    bichromatic_metricjoin(idxA::AbstractSearchIndex, ctx::AbstractContext, B::AbstractDatabase;
                            k::Int, rank::Int=1, q::Float64=0.9, mingroup::Int=8,
                            samedata::Bool=database(idxA) === B) -> Vector{Tuple{Int32,Int32,Float32}}

Metric (similarity) join between dataset `A` (indexed as `idxA`) and dataset `B`, when neither the
number of matches per `b` nor a join radius is known ahead of time. `k` is a deliberately overestimated
guess passed to a single [`searchbatch`](@ref) call; the real work is deciding, per candidate pair
`(a, b)`, whether it is close enough to actually count as a match -- i.e. picking a cutoff radius, and
picking it per `a` rather than a single global one, since different regions of `A` can have very
different local density.

The per-`a` radius is estimated from the *reverse* view of the very same `searchbatch` result: every
`b` that ranked `a` among its own closest `rank` candidates "votes" for `a` with its distance, and `a`'s
own cutoff is the `q`-quantile of the distances of everyone who voted for it. This is usually a far more
stable estimate than anything derivable from a single `b`'s own (possibly tiny/noisy) neighbor list,
since a well-connected `a` typically collects many more voters than `rank`. `a`'s that collect fewer
than `mingroup` voters (e.g. isolated points, or simply `length(B) < mingroup`) fall back to a single
global cutoff instead: the `q`-quantile of the pooled distances of every `a` that *did* reach
`mingroup` voters (free -- no extra distance evaluations, and on the right distance scale, unlike e.g.
a random-pair sample). If not even that pool has enough data (every group is under `mingroup`, a
pathological corner case), it falls back once more to a small random cross-sample of `A`-`B` pairs.
`mingroup` is clamped to at least `1` internally, since a genuinely empty group (zero voters, common
whenever `A` is larger than `B`, or `rank` doesn't reach every `a`) has no quantile to speak of.

If `database(idxA) === B` (i.e. `idxA` indexes `B` itself), self-matches (`a == b`) are excluded both
from voting and from the final result -- controlled by `samedata`, defaulting to that check, exactly as
in [`bichromatic_closestpair`](@ref). This matters more here than it does there: with the default
`rank == 1`, an unexcluded self-join would make *every* point's only rank-1 vote its own (trivial,
zero-distance) self-match, leaving no real neighborhood information in any group and collapsing every
threshold to the last-resort fallback.

Because the vote only uses `rank` (a small constant, `<< k`) candidates per `b`, but the final filter
is applied against every one of the `k` candidates `searchbatch` found, a single `b` can still end up
matched to several `a`'s -- this is a join, not a top-k query, so the output size is data-dependent, not
fixed.

# Arguments
- `idxA`: the search structure indexing dataset `A`
- `ctx`: the search context used by `idxA`
- `B`: the dataset queried against `idxA`, with no index of its own

# Keyword Arguments
- `k`: overestimated neighbor count for the initial `searchbatch(idxA, ctx, B, k)`; there is no good
  data-independent default, so this must be supplied
- `rank`: how many of each `b`'s top candidates vote for their respective `a` (`<< k` in practice,
  e.g. `1`-`3`)
- `q`: quantile used both per-group and for the pooled global fallback
- `mingroup`: minimum number of voters an `a` needs before its own quantile is trusted over the
  (pooled, or last-resort sampled) global fallback; clamped to at least `1`
- `samedata`: whether `idxA` indexes `B` itself, i.e. whether self-matches must be excluded

# Returns
A `Vector` of `(a, b, dist)` triples (identifier `a` of `idxA`, identifier `b` of `B`, their distance)
that survived the per-`a` cutoff -- unsorted, and of a size determined by the data, not by `k`.

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
A = MatrixDatabase(rand(Float32, 2, 10^3))
B = MatrixDatabase(rand(Float32, 2, 10^3))
GA = SearchGraph(dist, A)
ctx = SearchGraphContext()
index!(GA, ctx)

pairs = bichromatic_metricjoin(GA, ctx, B; k=16)
```
"""
function bichromatic_metricjoin(idxA::AbstractSearchIndex, ctx::AbstractContext, B::AbstractDatabase;
        k::Int,
        rank::Int=1,
        q::Float64=0.9,
        mingroup::Int=8,
        samedata::Bool=database(idxA) === B
    )::Vector{Tuple{Int32,Int32,Float32}}
    m, n = length(idxA), length(B)
    mingroup = max(mingroup, 1)
    ids, dists = searchbatch(idxA, ctx, B, k)

    groups = [Float32[] for _ in 1:m]
    @inbounds for j in 1:n, r in 1:min(rank, k)
        a = ids[r, j]
        (a == 0 || (samedata && a == j)) && continue
        push!(groups[a], dists[r, j])
    end

    globalfallback = metricjoin_globalfallback(groups, mingroup, q, idxA, B, samedata)

    threshold = Vector{Float32}(undef, m)
    @inbounds for a in 1:m
        g = groups[a]
        threshold[a] = length(g) >= mingroup ? quantile(g, q) : globalfallback
    end

    pairs = Tuple{Int32,Int32,Float32}[]
    @inbounds for j in 1:n, r in 1:k
        a = ids[r, j]
        (a == 0 || (samedata && a == j)) && continue
        d = dists[r, j]
        d <= threshold[a] && push!(pairs, (Int32(a), Int32(j), d))
    end

    pairs
end

"""
    metricjoin_globalfallback(groups, mingroup, q, idxA, B, samedata) -> Float32

The global cutoff used by [`bichromatic_metricjoin`](@ref) for `a`'s whose own group has fewer than
`mingroup` (already `>= 1`) voters: the `q`-quantile of the pooled distances of every group that does
reach `mingroup`, or -- only if no group anywhere does -- the `q`-quantile of a small random `A`-`B`
cross-sample (`samedata` makes this last-resort sample skip `a == b` pairs, which are trivially `0`
and would otherwise bias the estimate).
"""
function metricjoin_globalfallback(groups::Vector{Vector{Float32}}, mingroup::Int, q::Float64,
                                    idxA::AbstractSearchIndex, B::AbstractDatabase, samedata::Bool;
                                    samplesize::Int=64)
    pool = reduce(vcat, (g for g in groups if length(g) >= mingroup); init=Float32[])
    if !isempty(pool)
        return Float32(quantile(pool, q))
    end

    dist = distance(idxA)
    m, n = length(idxA), length(B)
    S = Vector{Float32}(undef, samplesize)
    for i in 1:samplesize
        u, v = rand(1:m), rand(1:n)
        while samedata && u == v
            v = rand(1:n)
        end

        S[i] = evaluate(dist, database(idxA, u), B[v])
    end

    Float32(quantile(S, q))
end
