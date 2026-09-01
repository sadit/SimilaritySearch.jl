# This file is a part of SimilaritySearch.jl

export rebuild

"""
    rebuild(g::SearchGraph, ctx::SearchGraphContext;
        progress=Progress(length(g); desc="rebuild", dt=2.0))

Rebuilds the `SearchGraph` index but seeing the whole dataset for the incremental construction, i.e.,
it can connect the i-th vertex to its knn in the 1..n possible vertices instead of its knn among 1..(i-1) as in the original algorithm.
Returns a new `SearchGraph` (the input `g` is not modified).

`g.algo[]`'s `maxvisits` is *not* carried over verbatim into the rebuild search: it may have
been tuned for a smaller, partial graph (e.g. by `OptimizeParameters` during incremental
insertion, before every vertex existed) or for a completely different distance/database
(e.g. a cheap proxy sketch used only to bootstrap `g`'s current topology), and in either case
isn't necessarily right for searching the *whole*, final graph with `g`'s actual distance,
which is what this function's own neighborhood search does. Carrying it over would silently
cap every node's rebuild-time search at that value, baking a permanently degraded topology
into the result -- no later `optimize_index!` call can fix that, since it only retunes
search-time parameters, not the graph's edges. `bsize`/`Δ` (the fields `optimize_index!`
itself explores) are kept as-is; only `maxvisits` is reset to a fresh `BeamSearch()`'s
default before searching, and the result carries that reset config forward too.

# Arguments

- `g`: The search index to be rebuild.
- `ctx`: The context to run the procedure, it can differ from the original one; `ctx.maxbatches`
  bounds the number of batches used by the internal [`@BATCHES`](@ref) calls (passed as
  `getminbatch(ctx, n)`), bounding the size of the per-batch scratch buffer (`qcache`) regardless
  of `n`; see [`getminbatch`](@ref) for the trade-offs of capping it.

# Keyword Arguments

- `progress`: a `ProgressMeter.Progress` object (or `nothing` to disable) used to report progress.

# Examples

```julia
ctx = SearchGraphContext()
G = SearchGraph(dist, db)
index!(G, ctx)
G = rebuild(G, ctx)
```
"""
function rebuild(g::SearchGraph, ctx::SearchGraphContext;
    progress=Progress(length(g); desc="rebuild", dt=2.0)
)
    n = length(g)
    ksearch = neighborhoodsize(ctx.neighborhood, n)
    @assert n > 0
    direct = Vector{Vector{UInt32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    minbatch = getminbatch(ctx, n)

    # see this function's docstring: `g` itself is left untouched (a temporary graph reuses
    # its db/adj/hints/dist, just with a freshly-capped `algo`) so the search below -- and
    # the `execute_callbacks!`-triggered auto-tuning after it, which is similarly anchored to
    # whatever `algo[]` it's handed (`runconfig`, src/searchgraph/optbs.jl) -- aren't
    # bottlenecked by a stale, possibly tiny `maxvisits` (see issue #59).
    search_algo = let bs = g.algo[]
        @set bs.maxvisits = BeamSearch().maxvisits
    end
    search_g = SearchGraph(distance(g), database(g), g.adj, g.hints, Ref(search_algo), Ref(n))

    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGIN
        # one private pair of scratch buffers per batch (`tmp`/`N`), indexed by @batchid() --
        # @nbatches() is bounded (~8 * nthreads(), via getminbatch), never by n, so this
        # never grows with the database size. Unlike Threads.threadid()-indexing, this
        # stays race-free under every scheduler (:static/:default/:greedy), not just the
        # default :static.
        qcache_ids   = zeros(UInt32,  ksearch, 2 * @nbatches())
        qcache_dists = zeros(Float32, ksearch, 2 * @nbatches())
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
        tmp = knnqueue(bctx, view(qcache_ids, 1:ksearch, 2 * @batchid() - 1), view(qcache_dists, 1:ksearch, 2 * @batchid() - 1))
        N   = knnqueue(bctx, view(qcache_ids, 1:ksearch, 2 * @batchid()),     view(qcache_dists, 1:ksearch, 2 * @batchid()))
    @LOOP for objID in 1:n
        reuse!(tmp)
        reuse!(N)
        find_neighborhood!(N, search_g, bctx, database(g, objID), tmp, 1:-1; hints=first(neighbors(g.adj, objID)))
        direct[objID] = collect(IdView(N))
        # @info length(direct[objID]) neighbors_length(g.adj, objID)

        progress !== nothing && next!(progress)
    end
    end

    adj = AdjList(direct)
    @BATCHES getminbatch(ctx, length(direct)) scheduler=ctx.scheduler for nodeID in eachindex(direct)
        connect_reverse_links!(adj, nodeID, neighbors(adj, nodeID)) do relID
            relID != nodeID
        end
    end

    G = SearchGraph(distance(g), database(g), adj, copy(g.hints), Ref(search_algo), Ref(length(g)))

    execute_callbacks!(G, ctx, force=true)

    G
end
