mutable struct KnnSorted{VEC<:AbstractVector} <: AbstractKnn
    const items::VEC
    sp::Int32
    ep::Int32
    maxlen::Int32
    costevals::Int32
    costblocks::Int32
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

@inline Base.length(res::KnnSorted) = res.ep - res.sp + 1

"""
    maxlength(res::KnnSorted)

The maximum allowed cardinality (the k of knnSorted)
"""
@inline maxlength(res::KnnSorted) = res.maxlen

@inline nearest(res::KnnSorted) = res.items[res.sp]
@inline frontier(res::KnnSorted) = res.items[res.ep]


function viewitems(res::KnnSorted)
    view(res.items, res.sp:res.ep)
end

"""
    sortitems!(res::KnnSorted)

Sort items and returns a view of the active items; this operations destroys the internal structure.
To recover the required structure just apply `reverse!` on the view.
"""
function sortitems!(res::KnnSorted)
    viewitems(res)
end

"""
    push_item!(res::KnnSorted, p::IdWeight)

Appends an item into the result set
"""
@inline function push_item!(res::KnnSorted, item::IdWeight)
    len = length(res)
    sp, ep = res.sp, res.ep

    if len < maxlength(res)
        if ep == length(res.items)
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
        sort_last_item!(WeightOrder, res.items, sp, ep)
        res.ep = ep
        return true
    end

    item.weight >= maximum(res) && return false
    @inbounds res.items[ep] = item
    sort_last_item!(WeightOrder, res.items, sp, ep)
    true
end

push_item!(res::KnnSorted, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
push_item!(res::KnnSorted, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

@inline function pop_min!(res::KnnSorted)
    sp = res.sp
    p = res.items[sp]
    res.sp = sp + one(sp)
    p
end

@inline function pop_max!(res::KnnSorted)
    ep = res.ep
    p = res.items[ep]
    res.ep = ep - one(ep)
    p
end

"""
    reuse!(res::XKnnSet, maxlen=length(res.items))

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::KnnSorted, maxlen=length(res.items))
    # @assert maxlen <= length(res.items)
    res.sp = 1
    res.ep = 0
    res.maxlen = maxlen
    res.costevals = 0
    res.costblocks = 0
    res
end
