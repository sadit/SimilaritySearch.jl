"""
    KnnHeap{VEC<:AbstractVector} <: AbstractKnn

A k-NN result container backed by a binary max-heap (ordered by [`DistOrder`](@ref)).
The root of the heap always holds the current farthest item, so once the container is
full a new candidate can be accepted or discarded in `O(1)` amortized time by comparing
it against the root, and inserted in `O(log k)` time.

# Fields
- `items::VEC`: backing storage for the heap entries (each an [`IdDist`](@ref)).
- `min::IdDist`: the closest item seen so far (tracked separately from the heap root).
- `len::Int32`: number of active items currently stored.
- `maxlen::Int32`: maximum number of items to keep (the `k` of the k-nn search).

Use [`knnqueue`](@ref) to create one instead of calling the constructor directly.

# Examples

```julia
res = knnqueue(KnnHeap, 3)  # k = 3
push_item!(res, 1, 0.5f0)
push_item!(res, 2, 0.1f0)
nearest(res)     # IdDist with the smallest distance seen so far
viewitems(res)   # view of the active items
```
"""
mutable struct KnnHeap{VEC<:AbstractVector} <: AbstractKnn
    items::VEC
    min::IdDist
    len::Int32
    maxlen::Int32
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
@inline frontier(res::KnnHeap) = res.items[1]

"""
    nearest(res::KnnHeap)

Returns the closest item ([`IdDist`](@ref)) seen so far in `res`.
"""
@inline nearest(res::KnnHeap) = res.min


"""
    viewitems(res::KnnHeap)

Returns a zero-copy view of the active items of `res` (in heap order, not sorted by
distance). Use [`sortitems!`](@ref) if a distance-sorted view is needed instead.
"""
function viewitems(res::KnnHeap)
    view(res.items, 1:res.len)
end

"""
    sortitems!(res::KnnHeap)

Sort items and returns a view of the active items; this operations destroys the internal heap structure.
It is possible to give the heap structure without calling `heapify!` just applying `reverse!` on the view.
"""
function sortitems!(res::KnnHeap)
    it = viewitems(res)
    heapsort!(DistOrder, it)
    it
end

"""
    push_item!(res::KnnHeap, p::IdDist)

Appends an item into the result set
"""
@inline function push_item!(res::KnnHeap, item::IdDist)
    len = res.len

    if length(res) < maxlength(res)
        len += one(len)
        res.items[len] = item
        heapfix_up!(DistOrder, res.items, len)
        if len == one(len) || lt(DistOrder, item, res.min)
            res.min = item
        end

        res.len = len
        return true
    end

    item.dist >= maximum(res) && return false
    res.items[1] = item
    heapfix_down!(DistOrder, res.items, len)
    if lt(DistOrder, item, res.min)
        res.min = item
    end

    true
end

"""
    push_item!(res::KnnHeap, i::Integer, d::Real)

Convenience overload of [`push_item!`](@ref) that builds the [`IdDist`](@ref) item from
an `id`/`dist` pair given as separate arguments.
"""
push_item!(res::KnnHeap, i::Integer, d::Real) = push_item!(res, IdDist(convert(UInt32, i), convert(Float32, d)))

"""
    push_item!(res::KnnHeap, p::Pair)

Convenience overload of [`push_item!`](@ref) that builds the [`IdDist`](@ref) item from
a `id => dist` pair.
"""
push_item!(res::KnnHeap, p::Pair) = push_item!(res, IdDist(convert(UInt32, p.first), convert(Float32, p.second)))

"""
    pop_max!(res::KnnHeap)

Removes and returns the farthest item (the heap root) from `res`, shrinking its length by one.
"""
@inline function pop_max!(res::KnnHeap)
    p = res.items[1]
    len = res.len
    heapswap!(res.items, 1, len)
    len -= 1
    heapfix_down!(DistOrder, res.items, len)
    res.len = len
    p
end

"""
    reuse!(res::KnnHeap, maxlen=length(res.items))

Resets `res` to a fresh initial state (empty, with capacity `maxlen`), reusing its
existing memory buffers instead of allocating a new result set.
"""
@inline function reuse!(res::KnnHeap, maxlen::Int=length(res.items))
    @assert maxlen <= length(res.items)
    res.min = zero(IdDist)
    res.len = 0
    res.maxlen = maxlen
    res
end

"""
    reuse!(res::KnnHeap{T}, items::T, maxlen=length(items)) where T

Like `reuse!(res, maxlen)`, but also replaces the backing storage of `res` with `items`
before resetting its state.
"""
@inline function reuse!(res::KnnHeap{T}, items::T, maxlen::Int=length(items)) where T
    res.items = items
    reuse!(res, maxlen)
end
