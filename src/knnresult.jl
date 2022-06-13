# This file is a part of SimilaritySearch.jl
export KnnResult
export maxlength, maxlength, getdist, getid, idview, distview, reuse!

struct IdDistArray
    id::Vector{Int32}
    dist::Vector{Float32}
    k::Int  # number of neighbors
end

@inline idview(a::IdDistArray) = a.id
@inline distview(a::IdDistArray) = a.dist
@inline getid(a::IdDistArray, i::Integer) = @inbounds a.id[i]
@inline getdist(a::IdDistArray, i::Integer) = @inbounds a.dist[i]
@inline maxlength(a::IdDistArray) = a.k
@inline Base.maximum(a::IdDistArray) = last(a.dist)
@inline Base.argmax(a::IdDistArray) = last(a.id)
@inline Base.minimum(a::IdDistArray) = first(a.dist)
@inline Base.argmin(a::IdDistArray) = first(a.id)
@inline Base.length(a::IdDistArray) = length(a.id)
@inline Base.firstindex(a::IdDistArray) = 1
@inline Base.lastindex(a::IdDistArray) = length(a)
@inline Base.first(a::IdDistArray) = @inbounds a[begin]
@inline Base.last(a::IdDistArray) = @inbounds a[end]
@inline Base.eachindex(a::IdDistArray) = eachindex(a.id)
@inline Base.popfirst!(a::IdDistArray) = popfirst!(a.id) => popfirst!(a.dist)
@inline Base.pop!(a::IdDistArray) = pop!(a.id) => pop!(a.dist)
@inline Base.getindex(a::IdDistArray, i::Integer) = @inbounds (a.id[i] => a.dist[i])

@inline function Base.push!(a::IdDistArray, p::Pair)
    push!(a.id, p.first)
    push!(a.dist, p.second)
    a
end

@inline function Base.setindex!(a::IdDistArray, p::Pair, i::Integer)
    @inbounds a.id[i] = p.first
    @inbounds a.dist[i] = p.second
end

@inline function Base.sizehint!(a::IdDistArray, sz::Integer)
    sizehint!(a.id, sz)
    sizehint!(a.dist, sz)
    a
end

function reuse!(a::IdDistArray, k::Integer)
    @assert k > 0
    empty!(a.id)
    empty!(a.dist)
    sizehint!(a, k)
    IdDistArray(a.id, a.dist, k)
end

"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResult{IdDistArray_} # <: AbstractVector{Tuple{IdType,DistType}}
    items::IdDistArray_
end

function KnnResult(k::Integer)
    @assert k > 0
    items = IdDistArray(Vector{Int32}(undef, 0), Vector{Float32}(undef, 0), k)
    sizehint!(items, k)
    KnnResult(items)
end

"""
    _shifted_fixorder!(res::KnnResult, sp, ep)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function _shifted_fixorder!(res::KnnResult, sp::Int, ep::Int)
    ep == sp && return
    id, dist = res.items.id, res.items.dist
    @inbounds i, d = id[ep], dist[ep]
    pos = _find_inspos(dist, sp, ep, d)
    _shift_vector(id, pos, ep, i)
    _shift_vector(dist, pos, ep, d)
    nothing
end

@inline function _find_inspos(dist::Vector{Float32}, sp::Int, ep::Int, d::Float32)
    @inbounds while (mid = ep-sp) > 16
        mid = sp + (mid >> 1)
        d < dist[mid] || break
        ep = mid
    end
    
    @inbounds while ep > sp
        ep -= 1
        d < dist[ep] || return ep + 1
    end

    ep
end

@inline function _shift_vector(arr::Vector, sp::Int, ep::Int, val)
    #=@inbounds while ep > sp;  arr[ep] = arr[ep-1]; ep -= 1; end=#
    unsafe_copyto!(arr, sp+1, arr, sp, ep-sp)
    @inbounds arr[sp] = val
end

"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, id::Integer, dist::Real; sp=1, k=maxlength(res))
    len = length(res)

    if len < k
        push!(res.items, id => dist)
        _shifted_fixorder!(res, sp, len+1)
        return true
    end

    dist >= maximum(res) && return false

    @inbounds res.items[end] = id => dist
    _shifted_fixorder!(res, sp, len)
    true
end

#@inline Base.push!(res::KnnResult, id::Integer, dist::Real) = push!(res, convert(Int32, id), convert(Float32, dist))
@inline Base.push!(res::KnnResult, p::Pair) = push!(res, p.first, p.second)

"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline Base.popfirst!(res::KnnResult) = popfirst!(res.items)

"""
    pop!(res::KnnResult)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline Base.pop!(res::KnnResult) = pop!(res.items)

"""
    maxlength(res::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult) = maxlength(res.items)
@inline Base.length(res::KnnResult) = length(res.items)

"""
    reuse!(res::KnnResult)
    reuse!(res::KnnResult, k::Integer)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline reuse!(res::KnnResult, k::Integer=maxlength(res)) = KnnResult(reuse!(res.items, k))

"""
    getindex(res::KnnResult, i)

Access the i-th item in `res`
"""
@inline Base.getindex(res::KnnResult, i::Integer) = @inbounds res.items[i]

@inline getid(res::KnnResult, i::Integer) = getid(res.items, i)
@inline getdist(res::KnnResult, i::Integer) = getdist(res.items, i)

@inline Base.last(res::KnnResult) = last(res.items)
@inline Base.first(res::KnnResult) = first(res.items)
@inline Base.maximum(res::KnnResult) = maximum(res.items)
@inline Base.minimum(res::KnnResult) = minimum(res.items)
@inline Base.argmax(res::KnnResult) = argmax(res.items)
@inline Base.argmin(res::KnnResult) = argmin(res.items)
@inline Base.firstindex(res::KnnResult) = firstindex(res.items)
@inline Base.lastindex(res::KnnResult) = lastindex(res.items)

@inline idview(res::KnnResult) = idview(res.items)
@inline distview(res::KnnResult) = distview(res.items)

@inline Base.eachindex(res::KnnResult) = eachindex(res.items)
Base.eltype(res::KnnResult) = Pair{Int32,Float32}

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