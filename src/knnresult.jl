# This file is a part of SimilaritySearch.jl
export KnnResult, KnnResultSet
export maxlength, getdist, getid, idview, distview, reuse!

###### KnnResult backends
### array implementation, can grow efficiently

abstract type AbstractIdDist end
struct IdDistArray <: AbstractIdDist
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
    _shifted_fixorder!(a::IdDistArray, sp, ep)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function _shifted_fixorder!(a, sp::Int, ep::Int)
    ep == sp && return
    @inbounds i, d = a[ep]
    pos = _find_inspos(a, sp, ep, d)
    _shift_vector(a.id, pos, ep, i)
    _shift_vector(a.dist, pos, ep, d)
    nothing
end

@inline function _find_inspos(a, sp::Int, ep::Int, d::Float32)
    while (mid = ep-sp) > 16
        mid = sp + (mid >> 1)
        d < getdist(a, mid) || break
        ep = mid
    end

    while ep > sp
        ep -= 1
        d < getdist(a, ep) || return ep + 1
    end

    ep
end

@inline function _shift_vector(arr::Vector, sp::Int, ep::Int, val)
    #=@inbounds while ep > sp;  arr[ep] = arr[ep-1]; ep -= 1; end=#
    unsafe_copyto!(arr, sp+1, arr, sp, ep-sp)
    @inbounds arr[sp] = val
end

############
## Matrix based: uses preallocated matrices of identifiers and distances, can't grow. IdDistViews should not be stored (instead save the original KnnResultSet)

struct KnnResultSet
    id::Matrix{Int32}
    dist::Matrix{Float32}
    len::Vector{Int}
end

function KnnResultSet(k::Integer, m::Integer)
    @assert k > 0 && m > 0
    KnnResultSet(Matrix{Int32}(undef, k, m), Matrix{Float32}(undef, k, m))
end

function KnnResultSet(id::Matrix{Int32}, dist::Matrix{Float32})
    KnnResultSet(id, dist, zeros(Int, size(id, 2)))
end

"""
    IdDistViews(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct IdDistViews <: AbstractIdDist
    parent::KnnResultSet
    i::Int
    id::Ptr{Int32}
    dist::Ptr{Float32}
end

@inline idview(a::IdDistViews) = @view a.parent.id[eachindex(a), a.i]
@inline distview(a::IdDistViews) = @view a.parent.dist[eachindex(a), a.i]

@inline getid(a::IdDistViews, i::Integer) = unsafe_load(a.id, i)
@inline getdist(a::IdDistViews, i::Integer) = unsafe_load(a.dist, i)
@inline maxlength(a::IdDistViews) = size(a.parent.id, 1)
@inline Base.length(a::IdDistViews) = @inbounds a.parent.len[a.i]
@inline Base.maximum(a::IdDistViews) = unsafe_load(a.dist, length(a))
@inline Base.argmax(a::IdDistViews) = unsafe_load(a.id, length(a))
@inline Base.minimum(a::IdDistViews) = unsafe_load(a.dist, 1)
@inline Base.argmin(a::IdDistViews) = unsafe_load(a.id, 1)
@inline Base.firstindex(a::IdDistViews) = 1
@inline Base.lastindex(a::IdDistViews) = length(a)
@inline Base.first(a::IdDistViews) = a[begin]
@inline Base.last(a::IdDistViews) = a[end]
@inline Base.eachindex(a::IdDistViews) = firstindex(a):lastindex(a)

@inline function Base.popfirst!(a::IdDistViews)
    p = argmin(a) => minimum(a)
    len = length(a) - 1
    unsafe_copyto!(a.id, a.id + sizeof(Int32), len)
    unsafe_copyto!(a.dist, a.dist + sizeof(Float32), len)
    a.parent.len[a.i] = len
    p
end

@inline function Base.pop!(a::IdDistViews)
    p = argmax(a) => maximum(a)
    a.parent.len[a.i] -= 1
    p
end

@inline Base.getindex(a::IdDistViews, i::Integer) = (getid(a, i) => getdist(a, i))

@inline function Base.push!(a::IdDistViews, p::Pair)
    l = length(a) + 1
    # check bounds? KnnResult.push! should check everything in fact, but...
    unsafe_store!(a.id, p.first, l)
    unsafe_store!(a.dist, p.second, l)
    a.parent.len[a.i] = l
    a
end

@inline function Base.setindex!(a::IdDistViews, p::Pair, i::Integer)
    # check bounds? KnnResult.push! should check everything in fact, but...
    unsafe_store!(a.id, p.first, i)
    unsafe_store!(a.dist, p.second, i)
end

@inline function Base.sizehint!(a::IdDistViews, sz::Integer)
    # do nothing
    a
end

function reuse!(a::IdDistViews, k::Integer)
    @assert k == maxlength(a) "IdDistViews can't change its size"
    a.parent.len[a.i] = 0
    a
end

@inline function _shift_vector(ptr::Ptr{T}, sp::Int, ep::Int, val) where T
    sp_ = sizeof(T) * (sp - 1)
    unsafe_copyto!(ptr + sp_ + sizeof(T), ptr + sp_, ep-sp)
    @inbounds unsafe_store!(ptr, val, sp)
end

#### Generic implementation of KnnResult
"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResult{IdDistArray_<:AbstractIdDist}
    items::IdDistArray_
end

function KnnResult(k::Integer)
    @assert k > 0
    items = IdDistArray(Vector{Int32}(undef, 0), Vector{Float32}(undef, 0), k)
    sizehint!(items, k)
    KnnResult(items)
end

function KnnResult(s::KnnResultSet, i::Integer)
    k = size(s.id, 1)
    sp = (i - 1) * k + 1
    KnnResult(IdDistViews(s, i, pointer(s.id, sp), pointer(s.dist, sp)))
end

"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, p::Pair{Int32,Float32}; sp=1, k=maxlength(res))
    len = length(res)

    if len < k
        push!(res.items, p)
        _shifted_fixorder!(res.items, sp, len+1)
        return true
    end

    p.second >= maximum(res) && return false

    @inbounds res.items[end] = p
    _shifted_fixorder!(res.items, sp, len)
    true
end

@inline Base.push!(res::KnnResult, id::Integer, dist::Real) = push!(res, convert(Int32, id) => convert(Float32, dist))

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