struct Knn{VEC<:AbstractVector} <: AbstractKnn
    items::VEC
    min::IdWeight
    len::Int32
    maxlen::Int32
    cost::Int32
    eblocks::Int32
end

@inline Base.length(res::Knn) = res.len

"""
    maxlength(res::Knn)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::Knn) = res.maxlen

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

    len = res.len
    min = res.min
    if p < maxlength(res)
        len += one(res.len)
        res.items[len] = item
        heapfix_up!(WeightOrder, res.items, len)
        if len == one(res.len) || lt(WeightOrder, item, min)
            min = item
        end

        return Knn(res.items, min, len, res.maxlen, res.cost, res.eblocks), true
    end

    item.weight >= maximum(res) && return res, false
    res.items[1] = item
    heapfix_down!(WeightOrder, res.items, len)
    if lt(WeightOrder, item, min)
        min = item
    end

    Knn(res.items, min, len, res.maxlen, res.cost, res.eblocks), true
end

push_item!(res::Knn, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
push_item!(res::Knn, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

"""
    reuse!(res::KnnSet, maxlen=length(res.items))

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::Knn, maxlen=length(res.items))
    @assert maxlen <= length(res.items)
    Knn(res.items, zero(IdWeight), zero(Int32), Int32(maxlen), zero(Int32), zero(Int32))
end
