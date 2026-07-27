"""
    KnnSorted{VEC<:AbstractVector} <: AbstractKnn

A k-NN result container that keeps its active items always sorted by distance (ascending,
[`DistOrder`](@ref)), using a bounded insertion sort on each push. It trades a slightly
higher insertion cost against [`KnnHeap`](@ref) for items that are always available in
sorted order without an explicit call to `sortitems!`.

# Fields
- `items::VEC`: backing storage for the entries (each an [`IdDist`](@ref)).
- `sp::Int32`: start position (index) of the active range within `items`.
- `ep::Int32`: end position (index) of the active range within `items`.
- `maxlen::Int32`: maximum number of items to keep (the `k` of the k-nn search).
- `costdist::Int32`: number of distance evaluations charged to this result set.
- `costblk::Int32`: number of block evaluations charged to this result set.

Use [`knnqueue`](@ref) to create one instead of calling the constructor directly.

# Examples

```julia
res = knnqueue(KnnSorted, 3)  # k = 3
push_item!(res, 1, 0.5f0)
push_item!(res, 2, 0.1f0)
nearest(res)     # closest item
viewitems(res)   # view of the active items, sorted by distance
```
"""
mutable struct KnnSorted{VEC<:AbstractVector} <: AbstractKnn
    items::VEC
    sp::Int32
    ep::Int32
    maxlen::Int32
    costdist::Int32
    costblk::Int32
end

"Number of distance evaluations charged to `res`."
@inline distance_evaluations(res::KnnSorted) = res.costdist
"Number of block evaluations charged to `res`."
@inline block_evaluations(res::KnnSorted) = res.costblk
"Adds `v` to the distance-evaluations counter of `res`."
@inline add_distance_evaluations!(res::KnnSorted, v) = (res.costdist += v)
"Adds `v` to the block-evaluations counter of `res`."
@inline add_block_evaluations!(res::KnnSorted, v) = (res.costblk += v)

"""
    sort_last_item!(order::Ordering, plist, sp, ep)

Sorts the last pushed item (at position `ep`) into its correct place within the active
range `sp:ep` of `plist`, in place. It implements insertion sort, which is efficient here
because the inserted item is expected to already be near its sorted position.
"""
@inline function sort_last_item!(order::Ordering, plist, sp, ep)
    sp == ep && return nothing # only one element, sorted
    @inbounds item = plist[ep]
    i = ep - 1
    @inbounds lt(order, plist[i], item) && return nothing # already sorted

    @inbounds while i >= sp
        p = plist[i]
        if lt(order, item, p)
            plist[i+1] = p
        else
            plist[i+1] = item
            return nothing
        end

        i -= 1
    end

    @inbounds plist[sp] = item
    nothing
end

#=@inline function sort_first_item!(order::Ordering, plist, sp, ep)
    # pos = sp
    @inbounds item = plist[sp]

    @inbounds while sp < ep && lt(order, item, plist[ep])
        ep -= one(ep)
    end

    @inbounds if sp < ep
        while sp < ep
            plist[sp] = plist[sp+1]
            sp += one(sp)
        end

        plist[sp] = item
    end

    nothing
end=#

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
@inline nearest(res::KnnSorted) = @inbounds res.items[res.sp]

"""
    frontier(res::KnnSorted)

Returns the farthest item ([`IdDist`](@ref)) currently stored in `res`, i.e., the item
that would be evicted next when a closer candidate is pushed.
"""
@inline frontier(res::KnnSorted) = @inbounds res.items[res.ep]


"""
    viewitems(res::KnnSorted)

Returns a zero-copy view of the active items of `res`, sorted by distance (ascending).
"""
@inline viewitems(res::KnnSorted) = view(res.items, res.sp:res.ep)

"""
    sortitems!(res::KnnSorted)

Sort items and returns a view of the active items; this operations destroys the internal structure.
To recover the required structure just apply `reverse!` on the view.
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
        if ep == length(res.items)  # reorganizing the queue (shift data to the beginning)
            i = zero(sp)
            for j in sp:ep
                i += one(sp)
                res.items[i] = res.items[j]
            end

            sp = res.sp = one(sp)
            ep = res.ep = i
        end

        ep += one(ep)
        res.items[ep] = item
        sort_last_item!(DistOrder, res.items, sp, ep)
        res.ep = ep
        return true
    end

    item.dist >= maximum(res) && return false
    @inbounds res.items[ep] = item
    sort_last_item!(DistOrder, res.items, sp, ep)
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
    @inbounds p = res.items[sp]
    res.sp = sp + one(sp)
    p
end

"""
    pop_max!(res::KnnSorted)

Removes and returns the farthest item from `res`, shrinking its active range from the end.
"""
@inline function pop_max!(res::KnnSorted)
    ep = res.ep
    @inbounds p = res.items[ep]
    res.ep = ep - one(ep)
    p
end

"""
    reuse!(res::KnnSorted, maxlen=length(res.items))

Resets `res` to a fresh initial state (empty, with capacity `maxlen`), reusing its
existing memory buffers instead of allocating a new result set.
"""
@inline function reuse!(res::KnnSorted, maxlen::Integer=length(res.items))
    # @assert maxlen <= length(res.items)
    res.sp = 1
    res.ep = 0
    res.maxlen = maxlen
    res.costdist = 0
    res.costblk = 0
    res
end

"""
    reuse!(res::KnnSorted{T}, items::T, maxlen=length(items)) where {T}

Like `reuse!(res, maxlen)`, but also replaces the backing storage of `res` with `items`
before resetting its state.
"""
@inline function reuse!(res::KnnSorted{T}, items::T, maxlen::Integer=length(items)) where {T}
    res.items = items
    reuse!(res, maxlen)
end
