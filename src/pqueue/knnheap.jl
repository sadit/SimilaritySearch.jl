mutable struct KnnHeap{VEC<:AbstractVector} <: AbstractKnn
    const items::VEC
    min::IdWeight
    len::Int32
    maxlen::Int32
    costevals::Int32
    costblocks::Int32
end

@inline Base.length(res::KnnHeap) = res.len

"""
    maxlength(res::KnnHeap)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnHeap) = res.maxlen
@inline frontier(res::KnnHeap) = res.items[1]
@inline nearest(res::KnnHeap) = res.min


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
    heapsort!(WeightOrder, it)
    it
end

"""
    push_item!(res::KnnHeap, p::IdWeight)

Appends an item into the result set
"""
@inline function push_item!(res::KnnHeap, item::IdWeight)
    len = res.len

    if length(res) < maxlength(res)
        len += one(len)
        res.items[len] = item
        heapfix_up!(WeightOrder, res.items, len)
        if len == one(len) || lt(WeightOrder, item, res.min)
            res.min = item
        end

        res.len = len
        return true
    end

    item.weight >= maximum(res) && return false
    res.items[1] = item
    heapfix_down!(WeightOrder, res.items, len)
    if lt(WeightOrder, item, res.min)
        res.min = item
    end

    true
end

push_item!(res::KnnHeap, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
push_item!(res::KnnHeap, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

@inline function pop_max!(res::KnnHeap)
    p = res.items[1]
    len = res.len
    heapswap!(res.items, 1, len)
    len -= 1
    heapfix_down!(WeightOrder, res.items, len)
    res.len = len
    p
end

"""
    reuse!(res::KnnSet, maxlen=length(res.items))

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::KnnHeap, maxlen=length(res.items))
    @assert maxlen <= length(res.items)
    res.min = zero(IdWeight)
    res.len = 0
    res.maxlen = maxlen
    res.costevals = 0
    res.costblocks = 0
    res
end
