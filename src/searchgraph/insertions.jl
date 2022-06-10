# This file is a part of SimilaritySearch.jl

"""
    append!(
        index::SearchGraph,
        db;
        neighborhood=Neighborhood(),
        parallel_block=1,
        parallel_minimum_first_block=parallel_block,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )

Appends all items in db to the index. It can be made in parallel or sequentially.

Arguments:

- `index`: the search graph index
- `neighborhood`: A [`Neighborhood`](@ref) object that specifies the kind of neighborhood that will be computed.
- `db`: the collection of objects to insert, an `AbstractDatabase` is the canonical input, but supports any iterable objects
- `parallel_block`: The number of elements that the multithreading algorithm process at once,
    it is important to be larger that the number of available threads but not so large since the quality of the search graph could degrade (a few times the number of threads is enough).
    If `parallel_block=1` the algorithm becomes sequential.
- `parallel_minimum_first_block`: The number of sequential appends before running parallel.
Note: Parallel doesn't trigger callbacks inside blocks.
- `callbacks`: A `SearchGraphCallbacks` object to be called after some insertions
    (specified by the `callbacks` object). These callbacks are used to maintain the algorithm
    in good shape after many insertions (adjust hyperparameters and the structure).
- `pools`: The set of caches used for searching.

Note 1: Callbacks are not executed inside parallel blocks
Note 2: Callbacks will be ignored if `callbacks=nothing`

"""
function Base.append!(
        index::SearchGraph,
        db;
        neighborhood=Neighborhood(),
        parallel_block=1,
        parallel_minimum_first_block=parallel_block,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )
    db = convert(AbstractDatabase, db)
    append!(index.db, db)

    parallel_block == 1 && return _sequential_append_loop!(index, neighborhood, callbacks, pools)

    n = length(index) + length(db)
    m = 0

    parallel_minimum_first_block = min(parallel_minimum_first_block, n)
    @inbounds while length(index) < parallel_minimum_first_block
        m += 1
        push_item!(index, db[m], neighborhood, false, callbacks, pools)
    end

    sp = length(index) + 1
    sp > n && return index

    resize!(index.links, n)
    _parallel_append_loop!(index, neighborhood, pools, sp, n, parallel_block, callbacks)
end

"""
    index!(index::SearchGraph; parallel_block=1, parallel_minimum_first_block=parallel_block, callbacks=SearchGraphCallbacks())

Indexes the already initialized database (e.g., given in the constructor method). It can be made in parallel or sequentially.
The arguments are the same than `append!` function but using the internal `index.db` as input.

"""
function index!(
        index::SearchGraph;
        neighborhood=Neighborhood(),
        parallel_block=1,
        parallel_minimum_first_block=parallel_block,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )
    @assert length(index) == 0 && length(index.db) > 0
    parallel_block == 1 && return _sequential_append_loop!(index, neighborhood, callbacks, pools)

    m = 0
    db = index.db
    n = length(db)

    parallel_minimum_first_block = min(parallel_minimum_first_block, n)
    @inbounds while length(index) < parallel_minimum_first_block
        m += 1
        push_item!(index, db[m], neighborhood, false, callbacks, pools)
    end

    sp = length(index) + 1
    sp > n && return index
    resize!(index.links, n)
    _parallel_append_loop!(index, neighborhood, pools, sp, n, parallel_block, callbacks)
end

function _sequential_append_loop!(index::SearchGraph, neighborhood::Neighborhood, callbacks, pools::SearchGraphPools)
    i = length(index)
    n = length(index.db)
    @inbounds while i < n
        i += 1
        push_item!(index, index.db[i], neighborhood, false, callbacks, pools)
    end

    index
end

function _connect_links(index, sp, ep)
    minbatch = getminbatch(0, ep-sp+1)
    @batch minbatch=minbatch per=thread for i in sp:ep
        @inbounds for id in index.links[i]
            lock(index.locks[id])
            try
                push!(index.links[id], i)
                # sat_should_push(index.links[id], index, index[i], i, -1.0) && push!(index.links[id], i)
            finally
                unlock(index.locks[id])
            end
        end
    end
end

function _parallel_append_loop!(index::SearchGraph, neighborhood::Neighborhood, pools::SearchGraphPools, sp, n, parallel_block, callbacks)
    while sp < n
        ep = min(n, sp + parallel_block)
        index.verbose && rand() < 0.01 && println(stderr, "appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())

        # searching neighbors
        # @show length(index.links), length(index.db), length(db), length(index.locks), length(index), sp, ep
        @batch minbatch=getminbatch(0, ep-sp+1) per=thread for i in sp:ep
            @inbounds index.links[i] = find_neighborhood(index, index.db[i], neighborhood, pools)
        end

        # connecting neighbors
        _connect_links(index, sp, ep)
        
        # increasing locks => new items are enabled for searching (and reported by length so they can also be hints)
        resize!(index.locks, ep)
        for i in sp:ep
            @inbounds index.locks[i] = Threads.SpinLock()
        end
        
        # apply callbacks
        callbacks !== nothing && execute_callbacks(callbacks, index, sp, ep)
        sp = ep + 1
    end

    index
end

"""
    push!(
        index::SearchGraph,
        item;
        neighborhood=Neighborhood(),
        push_item=true,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )

Appends an object into the index. It accepts the same arguments that `push!` but assuming some default values.

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `neighborhood`: A [`Neighborhood`](@ref) object that specifies the kind of neighborhood that will be computed.
- `push_item`: if `push_item=false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)
- `callbacks`: The set of callbacks that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- `pools`: The set of caches used for searching.

- Note: `callbacks=nothing` ignores the execution of any callback
"""
@inline function push!(
        index::SearchGraph,
        item;
        neighborhood=Neighborhood(),
        push_item=true,
        callbacks=SearchGraphCallbacks(),
        pools=getpools(index)
    )

    push_item!(index, item, neighborhood, push_item, callbacks, pools)
end

"""
    push_item!(
        index::SearchGraph,
        item,
        neighborhood,
        push_item,
        callbacks,
        pools
    )

Appends an object into the index

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `neighborhood`: A [`Neighborhood`](@ref) object that specifies the kind of neighborhood that will be computed.
- `push_item`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed).
- `callbacks`: The set of callbacks that are called whenever the index grows enough. Keeps hyperparameters and structure in shape.
- `pools`: The set of caches used for searching.

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
@inline function push_item!(
    index::SearchGraph,
    item,
    neighborhood::Neighborhood,
    push_item::Bool,
    callbacks::SearchGraphCallbacks,
    pools::SearchGraphPools
)
    neighbors = find_neighborhood(index, item, neighborhood, pools)
    push_neighborhood!(index, item, neighbors, callbacks; push_item)

    neighbors
end
