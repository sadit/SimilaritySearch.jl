struct XKnn{VEC<:AbstractVector} <: AbstractKnn
    items::VEC
    sp::Int32
    ep::Int32
    maxlen::Int32
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
        pos -= one(pos)
    end

    @inbounds if pos < ep
        while pos < ep
            plist[ep] = plist[ep-1]
            ep -= one(ep)
        end

        plist[ep] = item
    end

    nothing
end

function sort_first_item!(order::Ordering, plist, sp, ep)
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
end

@inline Base.length(res::XKnn) = res.ep - res.sp + 1

"""
    maxlength(res::XKnn)

The maximum allowed cardinality (the k of Xknn)
"""
@inline maxlength(res::XKnn) = res.maxlen

@inline nearest(res::XKnn) = res.items[res.sp]
@inline Base.maximum(res::XKnn) = res.items[res.ep].weight
@inline Base.argmax(res::XKnn) = res.items[res.ep].id
@inline Base.minimum(res::XKnn) = nearest(res).weight
@inline Base.argmin(res::XKnn) = nearest(res).id

@inline covradius(res::XKnn)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)

function viewitems(res::XKnn)
    view(res.items, res.sp:res.ep)
end

IdView(res::XKnn) = (res.items[i].id for i in res.sp:res.ep)
DistView(res::XKnn) = (res.items[i].weight for i in res.sp:res.ep)

"""
    sortitems!(res::XKnn)

Sort items and returns a view of the active items; this operations destroys the internal structure.
To recover the required structure just apply `reverse!` on the view.
"""
function sortitems!(res::XKnn)
    viewitems(res)
end

"""
    push_item!(res::XKnn, p::IdWeight)

Appends an item into the result set
"""
@inline function push_item!(res::XKnn, item::IdWeight)
    len = length(res)

    sp, ep = res.sp, res.ep
    if len < maxlength(res)
        if ep == length(res.items)
            i = zero(res.sp)
            for j in sp:ep
                i += one(res.sp)
                res.items[i] = res.items[j]
            end
            sp = one(res.sp)
            ep = i
        end

        ep += one(ep)
        
        res.items[ep] = item
        sort_last_item!(WeightOrder, res.items, sp, ep)
        return XKnn(res.items, sp, ep, res.maxlen, res.cost, res.eblocks), true
    end

    item.weight >= maximum(res) && return res, false
    @inbounds res.items[res.ep] = item
    sort_last_item!(WeightOrder, res.items, res.sp, res.ep)
    res, true
end

push_item!(res::XKnn, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
push_item!(res::XKnn, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

@inline function pop_min!(res::XKnn)
    p = res.items[res.sp]
    @reset res.sp += one(res.sp)
    res, p
end

@inline function pop_max!(res::XKnn)
    p = res.items[res.ep]
    @reset res.ep -= one(res.ep)
    res, p
end

"""
    reuse!(res::XKnnSet, maxlen=length(res.items))

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::XKnn, maxlen=length(res.items))
    # @assert maxlen <= length(res.items)
    XKnn(res.items, one(Int32), zero(Int32), Int32(maxlen), zero(Int32), zero(Int32))
end
