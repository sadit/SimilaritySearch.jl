# This file is a part of SimilaritySearch.jl

export rebuild

"""
    rebuild(g::SearchGraph, ctx::SearchGraphContext;
        progress=Progress(length(g); desc="rebuild", dt=2.0))

Rebuilds the `SearchGraph` index but seeing the whole dataset for the incremental construction, i.e.,
it can connect the i-th vertex to its knn in the 1..n possible vertices instead of its knn among 1..(i-1) as in the original algorithm.
Returns a new `SearchGraph` (the input `g` is not modified).

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
        find_neighborhood!(N, g, bctx, database(g, objID), tmp, 1:-1; hints=first(neighbors(g.adj, objID)))
        direct[objID] = collect(IdView(N))
        # @info length(direct[objID]) neighbors_length(g.adj, objID)

        progress !== nothing && next!(progress)
    end
    end

    directcount = Int32.(length.(direct))
    adj = AdjList(direct)
    @BATCHES getminbatch(ctx, length(direct)) scheduler=ctx.scheduler for nodeID in eachindex(direct)
        connect_reverse_links!(adj, nodeID, neighbors(adj, nodeID)) do relID
            relID != nodeID
        end
    end

    G = SearchGraph(distance(g), database(g), adj, copy(g.hints), Ref(g.algo[]), Ref(length(g)), directcount)

    execute_callbacks!(G, ctx, force=true)

    G
end
