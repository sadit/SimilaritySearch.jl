mutable struct XKnn{VEC<:AbstractVector} <: AbstractKnn
    items::VEC
    len::Int32
    cost::Int32
    eblocks::Int32
end

"""
    sort_last_item!(order::Ordering, plist)

Sorts the last push in place. It implements insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected to be really near of its sorted position)
"""
function sort_last_item!(order::Ordering, plist, sp, ep)
    pos = ep
    @inbounds item = plist[ep]

    @inbounds while pos > sp && lt(order, item, plist[pos-1])
        pos -= 1
    end

    @inbounds if pos < ep
        while pos < ep
            plist[ep] = plist[ep-1]
            ep -= 1
        end

        plist[ep] = item
    end

    nothing
end

function sort_first_item!(order::Ordering, plist, sp, ep)
    # pos = sp
    @inbounds item = plist[sp]

    @inbounds while sp < ep && lt(order, item, plist[ep])
        ep -= 1
    end

    @inbounds if sp < ep
        while sp < ep
            plist[sp] = plist[sp+1]
            sp += 1
        end

        plist[sp] = item
    end

    nothing
end

@inline Base.length(res::XKnn) = res.len

"""
    maxlength(res::XKnn)

The maximum allowed cardinality (the k of Xknn)
"""
@inline maxlength(res::XKnn) = length(res.items)

@inline nearest(res::XKnn) = res.items[res.len]
@inline Base.maximum(res::XKnn) = res.items[1].weight
@inline Base.argmax(res::XKnn) = res.items[1].id
@inline Base.minimum(res::XKnn) = nearest(res).weight
@inline Base.argmin(res::XKnn) = nearest(res).id

@inline covradius(res::XKnn)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)

function viewitems(res::XKnn)
    view(res.items, 1:res.len)
end

IdView(res::XKnn) = (res.items[i].id for i in 1:res.len)
DistView(res::XKnn) = (res.items[i].weight for i in 1:res.len)

"""
    sortitems!(res::XKnn)

Sort items and returns a view of the active items; this operations destroys the internal structure.
To recover the required structure just apply `reverse!` on the view.
"""
function sortitems!(res::XKnn)
    it = viewitems(res)
    reverse!(it)
    it
end

"""
    push_item!(res::XKnn, p::IdWeight)

Appends an item into the result set
"""
@inline function push_item!(res::XKnn, item::IdWeight)
    len = length(res)

    if len < maxlength(res)
        res.len += 1
        res.items[res.len] = item
        sort_last_item!(RevWeightOrder, res.items, 1, res.len)
        return true
    end

    item.weight >= maximum(res) && return false

    @inbounds res.items[1] = item
    sort_first_item!(RevWeightOrder, res.items, 1, res.len)
    true
end

push_item!(res::XKnn, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
push_item!(res::XKnn, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

@inline function pop_min!(res::XKnn)
    p = res.items[res.len]
    res.len -= 1
    p
end

@inline function pop_max!(res::XKnn)
    p = res.items[1]
    res.len -= 1
    for i in 1:res.len
        res.items[i] = res.items[i+1]
    end
    
    p
end

"""
    reuse!(res::XKnnSet)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::XKnn)
    res.len = 0
    res.cost = res.eblocks = 0
    res
end
