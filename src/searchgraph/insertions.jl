# This file is a part of SimilaritySearch.jl

"""
    append_items!(
        index::SearchGraph,
        ctx::SearchGraphContext,
        db
    )

Appends all items in db to the index. It can be made in parallel or sequentially.

# Arguments:

- `index`: the search graph index
- `db`: the collection of objects to insert, an `AbstractDatabase` is the canonical input, but supports any iterable objects
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).

# Examples

```julia
G = SearchGraph(dist, VectorDatabase())
ctx = SearchGraphContext()
append_items!(G, ctx, MatrixDatabase(rand(Float32, 8, 1000)))
```
"""
function append_items!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    items::AbstractDatabase;
)
    append_items!(index.db, items)
    index!(index, ctx)
end

"""
    add_inform_message(index::SearchGraph, sp, ep) -> String

The `add!` progress line, worded identically on both insertion paths (#66): the per-item
sequential path used to omit `n.size-quantiles`, so a reporter saw one schema under
`Threads.nthreads() > 1` and another under `-t 1`. The quantile is taken over the range just
inserted (`sp:ep`), not over the whole graph, so it stays O(ep - sp) -- a single value on the
sequential path. Called only from `@inform`'s thunk, i.e. never when the context is silent.
"""
function add_inform_message(index::SearchGraph, sp, ep)
    "add! sp=$sp ep=$ep $(index.algo[]) n.size-quantiles=$(quantile(neighbors_length.(Ref(index.adj), sp:ep), 0:0.25:1.0))"
end

function _sequential_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n, qcache_ids, qcache_dists, objects)
    @inbounds while sp <= n
        ksearch = neighborhoodsize(ctx.neighborhood, sp)
        tmp       = knnqueue(ctx, view(qcache_ids, 1:ksearch, 1), view(qcache_dists, 1:ksearch, 1))
        neighbors = knnqueue(ctx, view(qcache_ids, 1:ksearch, 2), view(qcache_dists, 1:ksearch, 2))

        push_item!(index, ctx, objects[sp], tmp, neighbors, false)
        sp += 1
    end
end

function _parallel_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n, qcache_ids, qcache_dists, objects)
    resize!(index.adj, n)

    while sp <= n
        # `sp` is a parameter reassigned at the bottom of this loop *and* captured by the
        # batch closure below, so Julia keeps it in a `Core.Box` and reading it yields `Any`.
        # Unwrap it once, here, before anything is derived from it: `ep` is computed from it,
        # and a single untyped endpoint is enough to make `objID` untyped inside the loop,
        # hence `item`, hence every distance find_neighborhood! computes over `blockrange` --
        # a million boxed `Float32`s per 4k-object build. Typing only the range's start does
        # nothing; `ep` has to be typed too, which is why this sits above it (see issue #56).
        spb = sp::Int
        ep = min(n, spb + ctx.parallel_block - 1)  # spb:ep has at most ctx.parallel_block elements
        ksearch = neighborhoodsize(ctx.neighborhood, ep)
        # qcache width is sized from ctx.maxbatches (see index!), derived from actual buffer size
        minbatch = getminbatch(ep - spb + 1; maxbatches=size(qcache_ids, 2) ÷ 2)
        twin = zeros(UInt32, ep - spb + 1)   # per block object: the twin it is a near duplicate of, or 0

        @BATCHES minbatch scheduler=ctx.scheduler begin
        @BEGINBATCH
            bctx = beginbatch(ctx, @batchid())
            tmp       = knnqueue(bctx, view(qcache_ids, 1:ksearch, 2 * @batchid() - 1), view(qcache_dists, 1:ksearch, 2 * @batchid() - 1))
            neighbors_ = knnqueue(bctx, view(qcache_ids, 1:ksearch, 2 * @batchid()),     view(qcache_dists, 1:ksearch, 2 * @batchid()))
        @LOOP for objID in spb:ep
            item = objects[objID]
            R = spb:objID-1
            reuse!(tmp)
            reuse!(neighbors_)
            find_neighborhood!(neighbors_, index, bctx, item, tmp, R)
            if isnearduplicate(bctx, neighbors_)
                twin[objID - spb + 1] = first(IdView(neighbors_))   # settled below, once the block is done
            else
                add!(index.adj, objID, IdView(neighbors_))
            end
        end
        end

        # near duplicates: resolved serially (twins inside the block may themselves be members
        # of each other), their adjacency emptied so the reverse links skip them, and attached
        # to their representatives afterwards
        settled = resolvemembers!(index, spb, ep, twin)
        OBSERVE(ctx, :add!, index, sp, ep)
        @inform ctx add_inform_message(index, sp, ep)
        # connecting neighbors
        connect_reverse_links!(index.adj, sp, ep; scheduler=ctx.scheduler)
        for (id, rep) in settled
            setmember!(index, id, rep)
        end
        index.len[] = ep

        # apply callbacks
        execute_callbacks!(index, ctx, sp, ep)
        sp = ep + 1
    end
end


"""
    index!(index::SearchGraph, ctx::SearchGraphContext)

Indexes the already initialized database (e.g., given in the constructor method). It can be made in parallel or sequentially.
The arguments are the same than `append_items!` function but using the internal `index.db` as input.

# Arguments:

- `index`: The graph index
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).

"""
index!(index::SearchGraph, ctx::SearchGraphContext) = _index!(index, ctx, database(index))

"Indexes the unindexed tail of the database, querying the graph with `objects[i]` for object `i` (see `InsertionSource`)."
function _index!(index::SearchGraph, ctx::SearchGraphContext, objects)
    n = length(database(index))
    @assert n > 0
    # one tuning pool for this insertion, drawn from the range it is about to fill, so every
    # callback scores on the same population instead of a fresh sample (see `tuningpool`)
    ctx = tuningpool(ctx, length(index) + 1, n)

    if ctx.parallel_block == 1 || Threads.nthreads() == 1
        qcache_ids, qcache_dists = let s = neighborhoodsize(ctx.neighborhood, n), t = 2
            isodd(s) && (s += 1)
            zeros(UInt32, s, t), zeros(Float32, s, t)
        end
        _sequential_append_items_loop!(index, ctx, length(index) + 1, n, qcache_ids, qcache_dists, objects)
    else
        qcache_ids, qcache_dists = let s = neighborhoodsize(ctx.neighborhood, n), t = 2 * ctx.maxbatches
            isodd(s) && (s += 1)
            zeros(UInt32, s, t), zeros(Float32, s, t)
        end
        _parallel_append_items_loop!(index, ctx, length(index) + 1, n, qcache_ids, qcache_dists, objects)
    end

    index
end

function index!(idx::SearchGraph, ctx::SearchGraphContext, kind::Symbol; kwargs...)
    index!(idx, ctx, Val(kind); kwargs...)
end

"""
    push_item!(
        index::SearchGraph,
        ctx::SearchGraphContext,
        item,
        neighbors_,
        tmp,
        push_db::Bool
    )

Appends a single object into the index, computing its neighborhood, connecting reverse
links, and running the registered callbacks. Low-level function used by the sequential and
parallel insertion loops (`append_items!`/`index!`).

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `neighbors_`: knnqueue used to store the computed neighborhood of `item`, later attached to the graph.
- `tmp`: knnqueue used as scratch space by the neighborhood computation.
- `push_db`: if `false`, `item` is not appended to `index.db` (used when `item` is already present in the database but not yet indexed).
"""
@inline function push_item!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    item,
    neighbors_,
    tmp,
    push_db::Bool
)
    push_db && push_item!(index.db, item)
    find_neighborhood!(neighbors_, index, ctx, item, tmp, 1:-1)
    n = Int32(index.len[] + 1)
    if isnearduplicate(ctx, neighbors_)
        # a member: the single edge to its representative, no reverse link, nothing to search
        # through; the twin is a node because the search never answers with members
        setmember!(index, n, representative(index, first(IdView(neighbors_))))
        OBSERVE(ctx, :add!, index, n, n)
        @inform ctx add_inform_message(index, n, n)
        index.len[] = n
        execute_callbacks!(index, ctx)
        return index
    end
    add!(index.adj, n, IdView(neighbors_))
    OBSERVE(ctx, :add!, index, n, n)
    @inform ctx add_inform_message(index, n, n)
    if n > 1
        connect_reverse_links!(index.adj, n, neighbors(index.adj, n))
        execute_callbacks!(index, ctx)
    end
    index.len[] = n
    index
end
