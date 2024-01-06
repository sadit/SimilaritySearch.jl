# This file is a part of SimilaritySearch.jl

"""
    append_items!(
        index::SearchGraph,
        db;
        setup=SearchGraphSetup(),
        pools=getpools(index)
    )

Appends all items in db to the index. It can be made in parallel or sequentially.

Arguments:

- `index`: the search graph index
- `neighborhood`: A [`Neighborhood`](@ref) object that specifies the kind of neighborhood that will be computed.
- `db`: the collection of objects to insert, an `AbstractDatabase` is the canonical input, but supports any iterable objects
- `pools`: The set of caches used for searching.

Note 1: Callbacks are not executed inside parallel blocks
Note 2: Callbacks will be ignored if `callbacks=nothing`

"""
function append_items!(
        index::SearchGraph,
        db;
        setup=SearchGraphSetup(),
        pools=getpools(index)
    )
    db = convert(AbstractDatabase, db)
    append_items!(index.db, db)

    setup.parallel_block == 1 && return _sequential_append_items_loop!(index, setup, pools)

    n = length(index) + length(db)
    m = 0

    parallel_first_block = min(setup.parallel_first_block, n)
    @inbounds while length(index) < parallel_first_block
        m += 1
        push_item!(index, db[m], setup, false, pools)
    end

    sp = length(index) + 1
    sp > n && return index

    _parallel_append_items_loop!(index, setup, pools, sp, n)
end

"""
    index!(
        index::SearchGraph;
        setup=SearchGraphSetup()
        pools=getpools(index)
    )

Indexes the already initialized database (e.g., given in the constructor method). It can be made in parallel or sequentially.
The arguments are the same than `append_items!` function but using the internal `index.db` as input.

"""
function index!(
        index::SearchGraph;
        setup=SearchGraphSetup(),
        pools=getpools(index)
    )
    @assert length(index) == 0 && length(index.db) > 0
    setup.parallel_block == 1 && return _sequential_append_items_loop!(index, setup, pools)

    m = 0
    db = database(index)
    n = length(db)

    parallel_first_block = min(setup.parallel_first_block, n)
    @inbounds while length(index) < parallel_first_block
        m += 1
        push_item!(index, db[m], setup, false, pools)
    end

    sp = length(index) + 1
    sp > n && return index

    _parallel_append_items_loop!(index, setup, pools, sp, n)
end

function _sequential_append_items_loop!(index::SearchGraph, setup::SearchGraphSetup, pools::SearchGraphPools)
    i = length(index)
    db = index.db
    n = length(db)
    @inbounds while i < n
        i += 1
        push_item!(index, db[i], setup, false, pools)
    end

    index
end

function _parallel_append_items_loop!(index::SearchGraph, setup::SearchGraphSetup, pools::SearchGraphPools, sp, n)
    adj = index.adj
    resize!(adj, n)

    while sp < n
        ep = min(n, sp + setup.parallel_block)
        #index.verbose && rand() < 0.01 && println(stderr, "appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())

        # searching neighbors 
	      @batch minbatch=getminbatch(0, ep-sp+1) per=thread for i in sp:ep
            # parallel_block values are pretty small, better to use @threads directly instead of @batch
            @inbounds adj.end_point[i] = find_neighborhood(index, database(index, i), setup.neighborhood, pools)
        end

        # connecting neighbors
        connect_reverse_links(index.adj, sp, ep)
        index.len[] = ep

        # apply callbacks
        execute_callbacks(setup, index, sp, ep)
        setup.logger !== nothing && LOG(setup.logger, append_items!, index, sp, ep, n)
        sp = ep + 1
    end

    index
end

"""
    push_item!(
        index::SearchGraph,
        item;
        setup=SearchGraphSetup(),
        push_item=true,
        pools=getpools(index)
    )

Appends an object into the index. It accepts the same arguments that `push_item!`
but assuming some default values.

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `neighborhood`: A [`Neighborhood`](@ref) object that specifies the kind of neighborhood that will be computed.
- `push_db`: if `push_db=false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)
- `callbacks`: The set of callbacks that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- `pools`: The set of caches used for searching.

"""
@inline function push_item!(
        index::SearchGraph,
        item;
        setup=SearchGraphSetup(),
        push_db=true,
        pools=getpools(index)
    )

    push_item!(index, item, setup, push_db, pools)
end

"""
    push_item!(
        index::SearchGraph,
        item,
        setup,
        push_item,
        pools
    )

Appends an object into the index

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `neighborhood`: A [`Neighborhood`](@ref) object that specifies the kind of neighborhood that will be computed.
- `push_db`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed).
- `callbacks`: The set of callbacks that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- `pools`: The set of caches used for searching.

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
@inline function push_item!(
    index::SearchGraph,
    item,
    setup::SearchGraphSetup,
    push_db::Bool,
    pools::SearchGraphPools
)
    push_db && push_item!(index.db, item)
    neighbors = find_neighborhood(index, item, setup.neighborhood, pools)
    add_vertex!(index.adj, neighbors)
    n = index.len[] = length(index.adj)
    if n > 1 
        connect_reverse_links(index.adj, n, neighbors)
        execute_callbacks(setup, index)
    end
    
    setup.logger !== nothing && LOG(setup.logger, push_item!, index, n)
    index
end
