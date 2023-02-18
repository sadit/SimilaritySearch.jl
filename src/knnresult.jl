# This file is a part of SimilaritySearch.jl
# export AbstractResult
export KnnResult
export covradius, maxlength, reuse!

"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResult
    items::Vector{IdWeight}
    k::Int  # number of neighbors
end

function KnnResult(k::Integer)
    @assert k > 0
    res = KnnResult(Vector{IdWeight}(undef, 0), k)
    sizehint!(res.items, k)
    res
end

"""
    push_item!(res::KnnResult, p::IdWeight)
    push_item!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function push_item!(res::KnnResult, item::IdWeight)
    len = length(res)

    if len < maxlength(res)
        push!(res.items, item)
        sort_last_item!(WeightOrder, res.items)
        return true
    end

    item.weight >= maximum(res) && return false

    @inbounds res.items[end] = item
    sort_last_item!(WeightOrder, res.items)
    true
end

@inline function push_item!(res::KnnResult, item::IdWeight, sp::Int, ep=maxlength(res))
    len = length(res)

    if len < ep
        push!(res.items, item)
        sort_last_item!(WeightOrder, view(res.items, sp:ep))
        return true
    end

    item.weight >= maximum(res) && return false

    @inbounds res.items[end] = item
    sort_last_item!(WeightOrder, sort_last_item!(WeightOrder, view(res.items, sp:ep)))
    true
end

@inline push_item!(res::KnnResult, id::Integer, dist::Real) = push_item!(res, IdWeight(id, dist))

@inline covradius(res::KnnResult)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)

"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResult)
    popfirst!(res.items)
end

"""
    pop!(res::KnnResult)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult)
    pop!(res.items)
end

"""
    maxlength(res::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult) = res.k
@inline Base.length(res::KnnResult) = length(res.items)

"""
    reuse!(res::KnnResult)
    reuse!(res::KnnResult, k::Integer)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::KnnResult, k::Integer=res.k)
    @assert k > 0
    empty!(res.items)
    if k > res.k
        sizehint!(res.items, k)
    end

    KnnResult(res.items, k)
end

"""
    getindex(res::KnnResult, i)

Access the i-th item in `res`
"""
@inline function Base.getindex(res::KnnResult, i)
    @inbounds res.items[i]
end

@inline Base.last(res::KnnResult) = last(res.items)
@inline Base.first(res::KnnResult) = @inbounds first(res.items)
@inline Base.maximum(res::KnnResult) = last(res.items).weight
@inline Base.minimum(res::KnnResult) = @inbounds first(res.items).weight
@inline Base.argmax(res::KnnResult) = last(res.items).id
@inline Base.argmin(res::KnnResult) = @inbounds first(res.items).id
@inline Base.firstindex(res::KnnResult) = 1
@inline Base.lastindex(res::KnnResult) = length(res.items)

@inline Base.eachindex(res::KnnResult) = firstindex(res):lastindex(res)
Base.eltype(res::KnnResult) = IdWeight

##### iterator interface
### KnnResult
"""
    Base.iterate(res::KnnResult, state::Int=1)

Support for iteration
"""
function Base.iterate(res::KnnResult, i::Int=1)
    n = length(res)
    (n == 0 || i > n) && return nothing
    @inbounds res[i], i+1
end