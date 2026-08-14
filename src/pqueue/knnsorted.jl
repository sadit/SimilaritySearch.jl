"""
    KnnSorted{IDS<:AbstractVector{UInt32}, DSTS<:AbstractVector{Float32}} <: AbstractKnnQueue

A k-NN result container that keeps its active items always sorted by distance (ascending,
[`DistOrder`](@ref)), using a bounded binary-search + block-shift on each push. It trades
a slightly higher insertion cost against [`KnnHeap`](@ref) for items that are always
available in sorted order without an explicit call to `sortitems!`.

# Fields
- `ids::IDS`: backing storage for the identifiers (`UInt32`).
- `dists::DSTS`: backing storage for the distances (`Float32`), parallel to `ids`.
- `sp::Int32`: start position (index) of the active range.
- `ep::Int32`: end position (index) of the active range.
- `maxlen::Int32`: maximum number of items to keep (the `k` of the k-nn search).

**Invariant**: `ids[sp:ep]` / `dists[sp:ep]` is always sorted in ascending order by
distance. `sort_last_item!` is the sole function responsible for maintaining this.

Use [`knnqueue`](@ref) to create one instead of calling the constructor directly.

# Examples

```julia
res = knnqueue(KnnSorted, 3)  # k = 3
push_item!(res, 1, 0.5f0)
push_item!(res, 2, 0.1f0)
nearest(res)     # closest item
viewitems(res)   # lazy view of the active items, sorted by distance
```
"""
mutable struct KnnSorted{IDS<:AbstractVector{UInt32},
                         DSTS<:AbstractVector{Float32}} <: AbstractKnnQueue
    ids::IDS
    dists::DSTS
    sp::Int32
    ep::Int32
    maxlen::Int32
end

function KnnSorted(ids::IDS, dists::DSTS; is_items=false) where {IDS, DSTS}
    @assert length(ids) == length(dists)
    if is_items
        len = length(ids)
        X = (ids, dists)
        heapify!(_lt_dist, _swap_ids_dists, X, len)
        heapsort!(_lt_dist, _swap_ids_dists, X, len)
        KnnSorted(ids, dists, one(Int32), Int32(len), Int32(len))
    else
        KnnSorted(ids, dists, one(Int32), zero(Int32), Int32(length(ids)))
    end
end

"""
    sort_last_item!(ids, dists, sp, ep)

Inserts the item at position `ep` into its correct sorted place within `ids[sp:ep]` /
`dists[sp:ep]`. Relies on the invariant that `ids[sp:ep-1]` / `dists[sp:ep-1]` is already
sorted in ascending order by distance.

The algorithm:
1. **Early exit**: if `dists[ep] >= dists[ep-1]` the array is already sorted.
2. **Binary search** on `dists[sp:ep-1]` to find the insertion point `lo` (first index
   where `dists[lo] > item_dist`).
3. **Block shift** via `copyto!` to move `ids[lo:ep-1] → ids[lo+1:ep]` (and likewise for
   `dists`), which the compiler/CPU can vectorize as a single `memmove`.
4. Write `item_id`/`item_dist` into position `lo`.
"""
@inline function sort_last_item!(ids, dists, sp, ep)
    sp == ep && return nothing                    # single element, already sorted
    @inbounds item_id   = ids[ep]
    @inbounds item_dist = dists[ep]
    @inbounds dists[ep - 1] <= item_dist && return nothing  # already in place

    # Binary search for the first position lo where dists[lo] > item_dist
    lo, hi = sp, ep - 1
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if dists[mid] <= item_dist
            lo = mid + 1
        else
            hi = mid
        end
    end

    # Block shift: move ids[lo:ep-1] → ids[lo+1:ep]
    # We use a manual reverse loop instead of `copyto!` because `copyto!` on overlapping
    # `SubArray`s falls back to an allocating path in Base Julia, causing massive GC pressure.
    # Nota: La asignación debe hacerse por separado para maximizar la optimización vectorial SIMD y caches

    @inbounds @simd for i in ep:-1:lo+1
        ids[i] = ids[i - 1]
    end
    @inbounds ids[lo] = item_id

    @inbounds @simd for i in ep:-1:lo+1
        dists[i] = dists[i - 1]
    end
    
    @inbounds dists[lo] = item_dist
    nothing
end

"Number of active items currently stored in `res`."
@inline Base.length(res::KnnSorted) = res.ep - res.sp + 1

"""
    maxlength(res::KnnSorted)

The maximum allowed cardinality (the k of knnSorted)
"""
@inline maxlength(res::KnnSorted) = res.maxlen

"""
    nearest(res::KnnSorted)

Returns the closest item ([`IdDist`](@ref)) currently stored in `res`.
"""
@inline nearest(res::KnnSorted) = @inbounds IdDist(res.ids[res.sp], res.dists[res.sp])

"""
    frontier(res::KnnSorted)

Returns the farthest item ([`IdDist`](@ref)) currently stored in `res`, i.e., the item
that would be evicted next when a closer candidate is pushed.
"""
@inline frontier(res::KnnSorted) = @inbounds IdDist(res.ids[res.ep], res.dists[res.ep])

"""
    viewitems(res::KnnSorted)

Returns a lazy zero-copy view of the active items of `res` as an `IdDistView` wrapper,
sorted by distance (ascending). Indexing returns `IdDist` pairs; iterating yields them in
order.
"""
@inline viewitems(res::KnnSorted) = IdDistView(res.ids, res.dists, Int(res.sp), Int(res.ep))

"""
    sortitems!(res::KnnSorted)

For `KnnSorted` items are always sorted; returns the `viewitems` view immediately.
"""
@inline sortitems!(res::KnnSorted) = viewitems(res)

"""
    push_item!(res::KnnSorted, p::IdDist)

Appends an item into the result set
"""
@inline function push_item!(res::KnnSorted, item::IdDist)
    len = length(res)
    sp, ep = res.sp, res.ep

    @inbounds if len < maxlength(res)
        if ep == length(res.ids)  # reorganizing the queue (shift data to the beginning)
            n = ep - sp + 1
            @inbounds @simd for i in 1:n
                res.ids[i] = res.ids[sp + i - 1]
                res.dists[i] = res.dists[sp + i - 1]
            end
            sp = res.sp = one(sp)
            ep = res.ep = Int32(n)
        end

        ep += one(ep)
        res.ids[ep]   = item.id
        res.dists[ep] = item.dist
        sort_last_item!(res.ids, res.dists, sp, ep)
        res.ep = ep
        return true
    end

    item.dist >= maximum(res) && return false
    @inbounds res.ids[ep]   = item.id
    @inbounds res.dists[ep] = item.dist
    sort_last_item!(res.ids, res.dists, sp, ep)
    true
end

"""
    push_item!(res::KnnSorted, i::Integer, d::Real)

Convenience overload of [`push_item!`](@ref) that builds the [`IdDist`](@ref) item from
an `id`/`dist` pair given as separate arguments.
"""
@inline push_item!(res::KnnSorted, i::Integer, d::Real) = push_item!(res, IdDist(convert(UInt32, i), convert(Float32, d)))

"""
    push_item!(res::KnnSorted, p::Pair)

Convenience overload of [`push_item!`](@ref) that builds the [`IdDist`](@ref) item from
a `id => dist` pair.
"""
@inline push_item!(res::KnnSorted, p::Pair) = push_item!(res, IdDist(convert(UInt32, p.first), convert(Float32, p.second)))

"""
    pop_min!(res::KnnSorted)

Removes and returns the closest item from `res`, shrinking its active range from the start.
"""
@inline function pop_min!(res::KnnSorted)
    sp = res.sp
    @inbounds p = IdDist(res.ids[sp], res.dists[sp])
    res.sp = sp + one(sp)
    p
end

"""
    pop_max!(res::KnnSorted)

Removes and returns the farthest item from `res`, shrinking its active range from the end.
"""
@inline function pop_max!(res::KnnSorted)
    ep = res.ep
    @inbounds p = IdDist(res.ids[ep], res.dists[ep])
    res.ep = ep - one(ep)
    p
end

"""
    reuse!(res::KnnSorted, maxlen=length(res.ids))

Resets `res` to a fresh initial state (empty, with capacity `maxlen`), reusing its
existing memory buffers instead of allocating a new result set.
"""
@inline function reuse!(res::KnnSorted, maxlen::Integer=length(res.ids))
    res.sp = 1
    res.ep = 0
    res.maxlen = maxlen
    res
end

"""
    reuse!(res::KnnSorted, ids, dists, maxlen=length(ids))

Like `reuse!(res, maxlen)`, but also replaces the backing storage of `res` with `ids`
and `dists` before resetting its state.
"""
@inline function reuse!(res::KnnSorted, ids::IDS, dists::DSTS, maxlen::Integer=length(ids)) where {IDS, DSTS}
    res.ids   = ids
    res.dists = dists
    reuse!(res, maxlen)
end
