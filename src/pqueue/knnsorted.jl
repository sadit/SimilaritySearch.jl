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

@inline Base.length(res::KnnSorted) = res.ep - res.sp + 1

"""
    maxlength(res::KnnSorted)

The maximum allowed cardinality (the k of knnSorted)
"""
@inline maxlength(res::KnnSorted) = res.maxlen

@inline nearest(res::KnnSorted) = @inbounds res.items[res.sp]
@inline frontier(res::KnnSorted) = @inbounds res.items[res.ep]


@inline viewitems(res::KnnSorted) = view(res.items, res.sp:res.ep)

"""
    sortitems!(res::KnnSorted)

Sort items and returns a view of the active items; this operations destroys the internal structure.
To recover the required structure just apply `reverse!` on the view.
"""
@inline sortitems!(res::KnnSorted) = viewitems(res)

"""
    push_item!(res::KnnSorted, p::IdWeight)

Appends an item into the result set
"""
@inline function push_item!(res::KnnSorted, item::IdWeight)
    len = length(res)
    sp, ep = res.sp, res.ep

    @inbounds if len < maxlength(res)
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

@inline push_item!(res::KnnSorted, i::Integer, d::Real) = push_item!(res, IdWeight(convert(UInt32, i), convert(Float32, d)))
@inline push_item!(res::KnnSorted, p::Pair) = push_item!(res, IdWeight(convert(UInt32, p.first), convert(Float32, p.second)))

@inline function pop_min!(res::KnnSorted)
    sp = res.sp
    @inbounds p = res.items[sp]
    res.sp = sp + one(sp)
    p
end

@inline function pop_max!(res::KnnSorted)
    ep = res.ep
    @inbounds p = res.items[ep]
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
