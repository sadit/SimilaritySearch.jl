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

"""
function append_items!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    db::AbstractDatabase;
)
    db = convert(AbstractDatabase, db)
    append_items!(index.db, db)

    ctx.parallel_block == 1 && return _sequential_append_items_loop!(index, ctx)

    n = length(index) + length(db)
    m = 0

    parallel_first_block = min(ctx.parallel_first_block, n)

    @inbounds while length(index) < parallel_first_block
        m += 1
        push_item!(index, ctx, db[m], false)
    end

    sp = length(index) + 1
    sp > n && return index

    _parallel_append_items_loop!(index, ctx, sp, n)
    index
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
    @assert length(index) == 0 && length(index.db) > 0
    ctx.parallel_block == 1 && return _sequential_append_items_loop!(index, ctx)

    m = 0
    db = database(index)
    n = length(db)

    parallel_first_block = min(ctx.parallel_first_block, n)
    @inbounds while length(index) < parallel_first_block
        m += 1
        push_item!(index, ctx, db[m], false)
    end

    sp = length(index) + 1
    sp > n && return index

    _parallel_append_items_loop!(index, ctx, sp, n)
    index
end

function _sequential_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext)
    i = length(index)
    db = index.db
    n = length(db)
    @inbounds while i < n
        i += 1
        push_item!(index, ctx, db[i], false)
    end

    index
end

function _parallel_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n)
    adj = index.adj
    resize!(adj, n)
    while sp <= n
        ep = min(n, sp + ctx.parallel_block)
        minbatch = getminbatch(ctx, ep - sp + 1)

        # searching neighbors 
        Threads.@threads :static for j in sp:minbatch:ep
            for i in j:min(ep, j + minbatch - 1)
                neighborhood = find_neighborhood(index, ctx, database(index, i))
                @inbounds adj.end_point[i] = collect(IdView(neighborhood))
            end
        end

        LOG(ctx.logger, :add_vertex!, index, ctx, sp, ep)
        # connecting neighbors
        connect_reverse_links(ctx.neighborhood, index.adj, sp, ep)
        index.len[] = ep

        # apply callbacks
        execute_callbacks(index, ctx, sp, ep)
        sp = ep + 1
    end
end

"""
    push_item!(
        index::SearchGraph,
        ctx,
        item;
        push_item=true
    )

Appends an object into the index.

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `push_db`: if `push_db=false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)

"""
@inline function push_item!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    item;
    push_db=true,
)

    push_item!(index, ctx, item, push_db)
end

"""
    push_item!(
        index::SearchGraph,
        ctx,
        item,
        push_item
    )

Appends an object into the index. Internal function

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `push_db`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed).

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
@inline function push_item!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    item,
    push_db::Bool
)
    push_db && push_item!(index.db, item)
    neighbors = find_neighborhood(index, ctx, item) |> IdView |> collect
    add_vertex!(index.adj, neighbors)
    n = index.len[] = length(index.adj)
    LOG(ctx.logger, :add_vertex!, index, ctx, n, n)
    if n > 1
        connect_reverse_links(ctx.neighborhood, index.adj, n, neighbors)
        execute_callbacks(index, ctx)
    end

    index
end
