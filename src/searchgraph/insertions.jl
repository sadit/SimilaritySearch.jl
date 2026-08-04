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
G = SearchGraph(; dist, db=VectorDatabase())
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

function _sequential_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n, qcache)
    @inbounds while sp <= n
        ksearch = neighborhoodsize(ctx.neighborhood, sp)
        tmp = knnqueue(ctx, view(qcache, 1:ksearch, 1))
        neighbors = knnqueue(ctx, view(qcache, 1:ksearch, 2))

        push_item!(index, ctx, database(index, sp), tmp, neighbors, false)
        sp += 1
    end
end

function _parallel_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n, qcache)
    resize!(index.adj, n)

    while sp <= n
        ep = min(n, sp + ctx.parallel_block - 1)  # sp:ep has at most ctx.parallel_block elements
        ksearch = neighborhoodsize(ctx.neighborhood, ep)
        # qcache's width is sized from ctx.maxbatches (see index!), so this cap is consistent
        # with it by construction; still derived from the buffer's actual size, not assumed.
        minbatch = getminbatch(ep - sp + 1; maxbatches=size(qcache, 2) ÷ 2)

        @BATCHES minbatch begin
        @BEGINBATCH
            bctx = @set ctx.batchid = @batchid()
            tmp = knnqueue(bctx, view(qcache, 1:ksearch, 2 * @batchid() - 1))
            neighbors_ = knnqueue(bctx, view(qcache, 1:ksearch, 2 * @batchid()))
        @LOOP for objID in sp:ep
            item = database(index, objID)
            R = sp:objID-1
            reuse!(tmp)
            reuse!(neighbors_)
            find_neighborhood!(neighbors_, index, bctx, item, tmp, R)
            add!(index.adj, objID, IdView(neighbors_))
        end
        end

        LOG(ctx.logger, :add!, index, ctx, sp, ep)
        # connecting neighbors
        connect_reverse_links!(index.adj, sp, ep)
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
function index!(index::SearchGraph, ctx::SearchGraphContext)
    n = length(database(index))
    @assert n > 0

    if ctx.parallel_block == 1 || Threads.nthreads() == 1
        qcache = let s = neighborhoodsize(ctx.neighborhood, n), t = 2
            isodd(s) && (s += 1)
            zeros(IdDist, s, t)
        end
        _sequential_append_items_loop!(index, ctx, length(index) + 1, n, qcache)
    else
        qcache = let s = neighborhoodsize(ctx.neighborhood, n), t = 2 * ctx.maxbatches
            isodd(s) && (s += 1)
            zeros(IdDist, s, t)
        end
        _parallel_append_items_loop!(index, ctx, length(index) + 1, n, qcache)
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
    add!(index.adj, n, IdView(neighbors_))
    LOG(ctx.logger, :add!, index, ctx, n, n)
    if n > 1
        connect_reverse_links!(index.adj, n, neighbors(index.adj, n))
        execute_callbacks!(index, ctx)
    end
    index.len[] = n
    index
end
