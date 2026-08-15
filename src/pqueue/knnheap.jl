"""
    KnnHeap{IDS<:AbstractVector{UInt32}, DSTS<:AbstractVector{Float32}} <: AbstractKnnQueue

A k-NN result container backed by a binary max-heap (ordered by [`DistOrder`](@ref)).
The root of the heap always holds the current farthest item, so once the container is
full a new candidate can be accepted or discarded in `O(1)` amortized time by comparing
it against the root, and inserted in `O(log k)` time.

# Fields
- `ids::IDS`: backing storage for the identifiers (`UInt32`).
- `dists::DSTS`: backing storage for the distances (`Float32`), parallel to `ids`.
- `min_id::UInt32`: the id of the closest item seen so far (tracked separately).
- `min_dist::Float32`: the distance of the closest item seen so far.
- `len::Int32`: number of active items currently stored.
- `maxlen::Int32`: maximum number of items to keep (the `k` of the k-nn search).

Use [`knnqueue`](@ref) to create one instead of calling the constructor directly.

# Examples

```julia
res = knnqueue(KnnHeap, 3)  # k = 3
push_item!(res, 1, 0.5f0)
push_item!(res, 2, 0.1f0)
nearest(res)     # IdDist with the smallest distance seen so far
IdDistView(res)  # view of the active items
```
"""
mutable struct KnnHeap{IDS<:AbstractVector{UInt32},
                       DSTS<:AbstractVector{Float32}} <: AbstractKnnQueue
    ids::IDS
    dists::DSTS
    min_id::UInt32
    min_dist::Float32
    len::Int32
    maxlen::Int32
end

function KnnHeap(ids::IDS, dists::DSTS; is_items=false) where {IDS, DSTS}
    @assert length(ids) == length(dists)
    if is_items
        len = length(ids)
        X = (ids, dists)
        heapify!(_lt_dist, _swap_ids_dists, X, len)
        
        min_id = ids[1]
        min_dist = dists[1]
        @inbounds @simd for i in 2:len
            d = dists[i]
            if d < min_dist
                min_dist = d
                min_id = ids[i]
            end
        end

        KnnHeap(ids, dists, min_id, min_dist, Int32(len), Int32(len))
    else
        KnnHeap(ids, dists, zero(UInt32), typemax(Float32), zero(Int32), Int32(length(ids)))
    end
end

"Number of active items currently stored in `res`."
@inline Base.length(res::KnnHeap) = res.len

"""
    maxlength(res::KnnHeap)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnHeap) = res.maxlen

"""
    frontier(res::KnnHeap)

Returns the farthest item currently stored in `res` (the heap root), i.e., the item
that would be evicted next when a closer candidate is pushed.
"""
@inline frontier(res::KnnHeap) = @inbounds IdDist(res.ids[1], res.dists[1])

"""
    nearest(res::KnnHeap)

Returns the closest item ([`IdDist`](@ref)) seen so far in `res`.
"""
@inline nearest(res::KnnHeap) = IdDist(res.min_id, res.min_dist)

"""
    sortitems!(res::KnnHeap)

Sort items and returns an `IdDistView` of the active items; this operation destroys the internal
heap structure. It is possible to restore the heap structure without calling `heapify!`
by applying `reverse!` on the returned view.
"""
function sortitems!(res::KnnHeap)
    heapsort!(_lt_dist, _swap_ids_dists, (res.ids, res.dists), Int(res.len))
    IdDistView(res)
end

"""
    push_item!(res::KnnHeap, p::IdDist)

Appends an item into the result set
"""
@inline function push_item!(res::KnnHeap, item::IdDist)
    len = res.len

    if length(res) < maxlength(res)
        len += one(len)
        @inbounds res.ids[len]   = item.id
        @inbounds res.dists[len] = item.dist
        heapfix_up!(_lt_dist, _swap_ids_dists, (res.ids, res.dists), len)
        if len == one(len) || item.dist < res.min_dist
            res.min_id   = item.id
            res.min_dist = item.dist
        end
        res.len = len
        return true
    end

    item.dist >= maximum(res) && return false
    @inbounds res.ids[1]   = item.id
    @inbounds res.dists[1] = item.dist
    heapfix_down!(_lt_dist, _swap_ids_dists, (res.ids, res.dists), len)
    if item.dist < res.min_dist
        res.min_id   = item.id
        res.min_dist = item.dist
    end
    true
end

"""
    push_item!(res::KnnHeap, i::Integer, d::Real)

Convenience overload of [`push_item!`](@ref) that builds the [`IdDist`](@ref) item from
an `id`/`dist` pair given as separate arguments.
"""
@inline push_item!(res::KnnHeap, i::Integer, d::Real) = push_item!(res, IdDist(convert(UInt32, i), convert(Float32, d)))

"""
    push_item!(res::KnnHeap, p::Pair)

Convenience overload of [`push_item!`](@ref) that builds the [`IdDist`](@ref) item from
a `id => dist` pair.
"""
@inline push_item!(res::KnnHeap, p::Pair) = push_item!(res, IdDist(convert(UInt32, p.first), convert(Float32, p.second)))

"""
    pop_max!(res::KnnHeap)

Removes and returns the farthest item (the heap root) from `res`, shrinking its length by one.
"""
@inline function pop_max!(res::KnnHeap)
    @inbounds p = IdDist(res.ids[1], res.dists[1])
    len = res.len
    _swap_ids_dists((res.ids, res.dists), 1, len)
    len -= 1
    heapfix_down!(_lt_dist, _swap_ids_dists, (res.ids, res.dists), len)
    res.len = len
    p
end

"""
    reuse!(res::KnnHeap, maxlen=length(res.ids))

Resets `res` to a fresh initial state (empty, with capacity `maxlen`), reusing its
existing memory buffers instead of allocating a new result set.
"""
@inline function reuse!(res::KnnHeap, maxlen::Int=length(res.ids))
    @assert maxlen <= length(res.ids)
    res.min_id   = zero(UInt32)
    res.min_dist = typemax(Float32)
    res.len      = 0
    res.maxlen   = maxlen
    res
end

"""
    reuse!(res::KnnHeap, ids, dists, maxlen=length(ids))

Like `reuse!(res, maxlen)`, but also replaces the backing storage of `res` with `ids`
and `dists` before resetting its state.
"""
@inline function reuse!(res::KnnHeap, ids::IDS, dists::DSTS, maxlen::Int=length(ids)) where {IDS, DSTS}
    res.ids   = ids
    res.dists = dists
    reuse!(res, maxlen)
end
