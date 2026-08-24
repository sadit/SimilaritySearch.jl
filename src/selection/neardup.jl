# This file is a part of SimilaritySearch.jl

export neardup

"""
    neardup(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, ϵ::Real; k::Int=8, blocksize::Int=256, filterblocks=true)
    neardup(dist::PreMetric, X::AbstractDatabase, ϵ::Real; recall=1.0, kwargs...)

Find near duplicates in database `X` using the empty index `idx`. The algorithm iteratively tries to index elements in `X`,
and items that are nearer than `ϵ` to some already indexed element are not inserted again (they are considered duplicates of it).

The two-argument `dist`-based method is a convenience wrapper that builds and manages its own index internally:
it uses an `ExhaustiveSearch` (exact) when `recall == 1.0`, or otherwise a `SearchGraph` (approximate) tuned to
approach the given `recall` via `OptimizeParameters(MinRecall(recall))`.

Returns a [`NearDupSelection`](@ref): the surviving objects as `centers`, which of them covers
each object of `X` as `assign`, and the index built over them as `idx`.

This is the radius-driven half of [`Selection`](@ref): you fix `ϵ` and the number of survivors is
whatever the data gives. The fixed-count selectors ([`fft`](@ref), [`dnet`](@ref),
[`randsel`](@ref), [`multirandsel`](@ref)) are the dual -- you fix the count and the radius falls
out -- and they report the same `centers`/`assign`/`assigndist` under the same names.

# Arguments
- `idx`: An empty index (e.g., a `SearchGraph` or an `ExhaustiveSearch`) -- only for the `idx`-based method
- `ctx`: the index's context (caches, hyperparameters, logger, etc) -- only for the `idx`-based method
- `dist`: the distance function to use -- only for the `dist`-based method
- `X`: The input dataset
- `ϵ`: the radius below which two objects count as duplicates of each other. It must not be
  negative -- every distance would exceed it, so nothing would ever be collapsed -- and a negative
  value is rejected rather than silently returning every object as its own center. `ϵ = 0` is
  meaningful: it collapses exact duplicates only. To pick one from the data rather than by hand,
  sample the distance distribution first with [`distsample`](@ref) and take a low quantile of it:

  ```julia
  ϵ = quantile(distsample(dist, X; samplesize=2^10), 0.01)
  D = neardup(dist, X, ϵ)
  ```

# Keyword Arguments
- `k`: The number of nearest neighbors to retrieve (some algorithms benefit from retrieving larger `k` values)
- `blocksize`: the number of items processed at a time
- `filterblocks`: if true then it filters neardups inside blocks (see `blocksize` parameter), otherwise, it supposes that blocks are free of neardups (e.g., randomized order).
- `recall`: (only for the `dist`-based method) target recall used to decide between an exact (`recall=1.0`) or approximate index

The `ctx`-based method reports through `ctx`: `verbose(ctx)` decides whether its progress messages
are produced, `ctx.reporters` where they go. The `dist`-based wrapper takes `verbose`, `reporters`
and `observers` directly, since it builds the context itself.

# Notes
- The index `idx` must support incremental construction
- If you need to customize object insertions, you must wrap the index `idx` and implement your custom methods; it requires valid implementations of the following functions:
   - `searchbatch(idx::AbstractSearchIndex, ctx, queries::AbstractDatabase, knns::Matrix, dists::Matrix)`
   - `distance(idx::AbstractSearchIndex)`
   - `length(idx::AbstractSearchIndex)`
   - `append_items!(idx::AbstractSearchIndex, ctx, items::AbstractDatabase)`
- The ``ϵ``-net itself is `centers`; `database(idx)` holds the same objects, in the same order

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 10^3))
ϵ = 0.1

# using an explicit index
G = SearchGraph(dist, VectorDatabase(Vector{Float32}[]))
ctx = SearchGraphContext()
D = neardup(G, ctx, X, ϵ; blocksize=256)
D.centers                     # the ϵ-net: which objects of X survived
D.centers[D.assign[7]]        # which of them covers object 7
D.assigndist[7]               # how far object 7 sits from it (<= ϵ)
D.covering, D.epsilon         # the radius actually needed, and the one asked for
D.costdists, D.costblocks     # cost of this call

# convenience wrapper (builds its own exact index since recall=1.0)
D2 = neardup(dist, X, ϵ)
```
"""
function neardup(dist::PreMetric, X::AbstractDatabase, ϵ::Real; recall=1.0,
        verbose::Bool=false, reporters=InformativeLog(), observers=nothing, kwargs...)
    dist_ = SimilaritySearch.Dist.Hacks.DistanceWithIdentifiers(dist, X)
    X_ = VectorDatabase(Int32[])
    if recall < 1.0
        idx = SearchGraph(dist_, X_)
        hyperparameters_callback = OptimizeParameters(MinRecall(recall))
        ctx = SearchGraphContext(; hyperparameters_callback, verbose, reporters, observers)
    else
        idx = ExhaustiveSearch(dist_, X_)
        ctx = GenericContext(; verbose, reporters, observers)
    end

    neardup(idx, ctx, VectorDatabase(UnitRange{Int32}(1, length(X))), ϵ; kwargs...)
end

function neardup(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, ϵ::Real;
    k::Int=8, blocksize::Int=256, filterblocks=true)

    ϵ >= 0 || throw(ArgumentError("neardup needs a non-negative ϵ, got $ϵ; see its docstring for how to estimate one with distsample"))
    ϵ = convert(Float32, ϵ)
    n = length(X)
    n == 0 && return NearDupSelection(idx, UInt32[], UInt32[], Float32[], 0f0, ϵ, 0, 0)

    blocksize = min(blocksize, n)
    knns_ids   = Matrix{UInt32}(undef, k, blocksize)
    knns_dists = Matrix{Float32}(undef, k, blocksize)

    # `L[i]` is the position in `M` of the center covering object `i`, never the center's own
    # identifier -- see `AbstractSelection`
    L = zeros(UInt32, n)
    D = zeros(Float32, n)
    M = UInt32[]
    imap = UInt32[]
    tmp = UInt32[]
    tmppos = UInt32[]

    dist_snapshot = copy(ctx.costdists)
    blk_snapshot = copy(ctx.costblocks)

    for range in Iterators.partition(1:n, blocksize)
        if length(idx) == 0
            verbose(ctx) && @inform ctx "neardup> starting: $(range), current elements: $(length(idx)), n: $n, ϵ: $ϵ"
            neardup_block!(idx, ctx, X, range, tmp, tmppos, L, D, M, ϵ; filterblocks)
        else
            empty!(imap)
            if size(knns_ids, 2) != length(range)
                rng = 1:length(range)
                knns_ids_   = view(knns_ids,   :, rng)
                knns_dists_ = view(knns_dists, :, rng)
                fill!(knns_ids_,   zero(UInt32))
                fill!(knns_dists_, typemax(Float32))
                searchbatch!(idx, ctx, X[range], knns_ids_, knns_dists_; sorted=true)
                knns_ids_view, knns_dists_view = knns_ids_, knns_dists_
            else
                fill!(knns_ids,   zero(UInt32))
                fill!(knns_dists, typemax(Float32))
                searchbatch!(idx, ctx, X[range], knns_ids, knns_dists; sorted=true)
                knns_ids_view, knns_dists_view = knns_ids, knns_dists
            end
            verbose(ctx) && @inform ctx "neardup> range: $(range), current elements: $(length(idx)), n: $n, ϵ: $ϵ"

            for (i, j) in enumerate(range) # collecting non-discarded near duplicated objects
                pid  = knns_ids_view[1, i]
                pdist = knns_dists_view[1, i]
                if pdist > ϵ
                    push!(imap, j)
                else
                    # `pid` is a position into `idx`, which is built in `M`'s order, so it is
                    # already the position into `centers` that `assign` wants
                    D[j] = pdist
                    L[j] = pid
                end
            end

            if length(imap) > 0
                neardup_block!(idx, ctx, X, imap, tmp, tmppos, L, D, M, ϵ; filterblocks)
            end
        end
    end
    
    verbose(ctx) && @inform ctx "neardup> finished current elements: $(length(idx)), n: $n, ϵ: $ϵ"

    costdists = distance_evaluations(ctx, dist_snapshot)
    costblocks = block_evaluations(ctx, blk_snapshot)

    NearDupSelection(idx, M, L, D, maximum(D), ϵ, costdists, costblocks)
end


"""
    neardup_block!(idx, ctx, X, imap, tmp, L, D, M, ϵ; filterblocks::Bool)

# Arguments:
- `idx` the output index
- `ctx` the index's context
- `X` input database- `L` nearest neighbors of the input database to non-near dups
- `imap` list of items to test and insert
- `tmp` a temporary buffer to save imap elements
- `tmppos` parallel to `tmp`: the position in `M` of each of its entries
- `L` for each object of the input database, the position in `M` of the center covering it
- `D` nearest neighbors distances of the input database to non-near dups
- `M` maps of `idx` to the input database
- `ϵ` radius to consider objects as near dups
- `filterblocks` if true it performs neardup in blocks
"""
function neardup_block!(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, imap, tmp, tmppos, L, D, M, ϵ; filterblocks::Bool)
    if !filterblocks
        append_items!(idx, ctx, X[imap])
        for i in imap
            push!(M, i)
            L[i] = length(M)   # a center is covered by itself, at its own position
            D[i] = 0.0f0
        end

        return
    end

    empty!(tmp)
    empty!(tmppos)
    n = length(imap)
    i = first(imap)
    push!(tmp, i)
    push!(M, i)
    push!(tmppos, length(M))
    L[i] = length(M)
    D[i] = 0.0f0

    dist = distance(idx)
    res = knnqueue(ctx, 1)

    for ii in 2:n
        reuse!(res)
        i = imap[ii]
        u = X[i]
        minbatch = getminbatch(ctx, length(tmp))

        @BATCHES minbatch scheduler=ctx.scheduler begin
            @BEGIN
                B_j = Vector{Int32}(undef, @nbatches())
                B_d = Vector{Float32}(undef, @nbatches())
            @BEGINBATCH
                b_min_j = zero(Int32)
                b_min_d = typemax(Float32)
            @LOOP for jj in firstindex(tmp):lastindex(tmp)
                # the winner is reported as its index into `tmp`, not as the object's own
                # identifier: `tmppos` turns it into the position in `M` that `L` records
                d = evaluate(dist, u, X[tmp[jj]])
                if d < b_min_d
                    b_min_d = d
                    b_min_j = jj
                end
            end
            @ENDBATCH
                B_j[@batchid] = b_min_j
                B_d[@batchid] = b_min_d
            @END
                for b in 1:@nbatches()
                    if B_j[b] > 0
                        push_item!(res, B_j[b], B_d[b])
                    end
                end
                add_distance_evaluations!(ctx, length(tmp))
        end

        let nn = nearest(res)
            if nn.dist > ϵ
                push!(tmp, i)
                push!(M, i)
                push!(tmppos, length(M))
                L[i] = length(M)
                D[i] = 0.0f0
            else
                L[i] = tmppos[nn.id]
                D[i] = nn.dist
            end
        end
    end

    append_items!(idx, ctx, X[tmp])
end
