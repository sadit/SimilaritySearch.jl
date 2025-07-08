mutable struct Knn{VEC<:AbstractVector} <: AbstractKnn
    items::VEC
    min::IdWeight
    len::Int32
    cost::Int32
    eblocks::Int32
end

@inline Base.length(res::Knn) = res.len

"""
    maxlength(res::Knn)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::Knn) = length(res.items)

#@inline Base.last(res::Knn) = res.items[1]
#@inline Base.first(res::Knn) = res.items[res.minpos]
@inline Base.maximum(res::Knn) = res.items[1].weight
@inline Base.argmax(res::Knn) = res.items[1].id
@inline nearest(res::Knn) = res.min
@inline Base.minimum(res::Knn) = nearest(res).weight
@inline Base.argmin(res::Knn) = nearest(res).id

@inline covradius(res::Knn)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)

function viewitems(res::Knn)
    view(res.items, 1:res.len)
end

IdView(res::Knn) = (res.items[i].id for i in 1:res.len)
DistView(res::Knn) = (res.items[i].weight for i in 1:res.len)


"""
    sortitems!(res::Knn)

Sort items and returns a view of the active items; this operations destroys the internal heap structure.
It is possible to give the heap structure without calling `heapify!` just applying `reverse!` on the view.
"""
function sortitems!(res::Knn)
    it = viewitems(res)
    heapsort!(WeightOrder, it)
    it
end

"""
    push_item!(res::Knn, p::IdWeight)

Appends an item into the result set
"""
@inline function push_item!(res::Knn, item::IdWeight)
    p = length(res)

    if p < maxlength(res)
        res.len += 1
        res.items[res.len] = item
        heapfix_up!(WeightOrder, res.items, res.len)
        if res.len == 1 || lt(WeightOrder, item, res.min)
            res.min = item
        end
        return true
    end

    item.weight >= maximum(res) && return false
    res.items[1] = item
    heapfix_down!(WeightOrder, res.items, res.len)
    if lt(WeightOrder, item, res.min)
        res.min = item
    end
    true
end

push_item!(res::Knn, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
push_item!(res::Knn, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

"""
    reuse!(res::KnnSet)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::Knn)
    res.len = 0
    res.min = IdWeight(0, 0f0)
    res.cost = res.eblocks = 0
    res
end
