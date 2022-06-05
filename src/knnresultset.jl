# This file is a part of SimilaritySearch.jl

export KnnResultSet, KnnResultView

struct KnnResultSet
    id::Matrix{Int32}
    dist::Matrix{Float32}
    len::Vector{Int32}
end

function KnnResultSet(k::Integer, m::Integer)
    @assert k > 0 && m > 0

    KnnResultSet(
        Matrix{Int32}(undef, k, m),
        Matrix{Float32}(undef, k, m),
        zeros(Int32, m)
    )
end

Base.size(knns::KnnResultSet) = size(knns.id)
Base.size(knns::KnnResultSet, dim) = size(knns.id, dim)

"""
    KnnResultView(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResultView <: AbstractKnnResult
    parent::KnnResultSet
    i::Int
    id::Ptr{Int32}
    dist::Ptr{Float32}

    function KnnResultView(parent::KnnResultSet, i::Integer)
        k = size(parent, 1)
        p = 1 + (i-1)*k
        new(parent, i, pointer(parent.id, p), pointer(parent.dist, p))
    end
end

"""
    reuse!(res::KnnResultView)
    reuse!(knns::KnnResultSet, i::Integer)

Resets `res` to an empty state
"""
@inline function reuse!(res::KnnResultView)
    res.parent.len[res.i] = 0
    res
end

@inline function reuse!(knns::KnnResultSet, i::Integer)
    knns.len[i] = 0
    KnnResultView(knns, i)
end

### push functions

"""
    push!(res::KnnResultView, item::Pair)
    push!(res::KnnResultView, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultView, id::Integer, dist::Real)
    k = length(res)

    if k < maxlength(res)
        k += 1
        unsafe_store!(res.id, id, k)
        unsafe_store!(res.dist, dist, k)
        @inbounds res.parent.len[res.i] = k
    
        _shifted_fixorder!(res, 1, k)
        return true
    end

    dist >= maximum(res) && return false

    unsafe_store!(res.id, id, k)
    unsafe_store!(res.dist, dist, k)
    _shifted_fixorder!(res, 1, k)
    true
end

"""
    _shifted_fixorder!(res::KnnResultView, sp, ep)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function _shifted_fixorder!(res::KnnResultView, sp::Int, ep::Int)
    id, dist = res.id, res.dist
    @inbounds i, d = unsafe_load(id, ep), unsafe_load(dist, ep)
    pos = _find_inspos(dist, sp, ep, d)
    _shift_vector(id, pos, ep, i)
    _shift_vector(dist, pos, ep, d)

    nothing
end

@inline function _find_inspos(dist::Ptr{Float32}, sp::Int, ep::Int, d::Float32)
    @inbounds while ep > sp
        ep -= 1
        d < unsafe_load(dist, ep) || return ep + 1
    end

    ep
end

@inline function _shift_vector(arr::Ptr, sp::Int, ep::Int, val)
    while ep > sp
        unsafe_store!(arr, unsafe_load(arr, ep-1), ep)
        ep -= 1
    end

    unsafe_store!(arr, val, ep)
end

#@inline Base.push!(res::KnnResultView, id::Integer, dist::Real) = push!(res, convert(Int32, id), convert(Float32, dist))
@inline Base.push!(res::KnnResultView, p::Pair) = push!(res, p.first, p.second)

### pop functions

"""
    popfirst!(p::KnnResultView)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResultView)
    @inbounds begin
        n = res.parent.len[res.i]
        res.parent.len[res.i] = n - 1
        _popfirst!(res.id, n) =>  _popfirst!(res.dist, n)
    end
end

@inline function _popfirst!(M::Ptr, len::Integer)
    @inbounds begin
        s = unsafe_load(M, 1)
        for i in 1:len-1
            unsafe_store!(M, unsafe_load(M, i+1), i)
        end

        s
    end
end

"""
    pop!(res::KnnResultView)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResultView)
    @inbounds begin
        n = res.parent.len[res.i]
        res.parent.len[res.i] = n - 1
        res[n]
    end
end

##### access functions #######

@inline getid(res::KnnResultView, i) = unsafe_load(res.id, i)
@inline getdist(res::KnnResultView, i) = unsafe_load(res.dist, i)

"""
    getindex(res::KnnResultView, i)

Access the i-th item in `res`
"""
@inline function Base.getindex(res::KnnResultView, i::Integer)
    @inbounds getid(res, i) => getdist(res, i)
end

"""
    maxlength(res::KnnResultView)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultView) = size(res.parent.id, 1)
@inline Base.length(res::KnnResultView) = @inbounds Int(res.parent.len[res.i])

@inline Base.last(res::KnnResultView) = argmax(res) => maximum(res)
@inline Base.first(res::KnnResultView) = argmin(res) => minimum(res)
@inline Base.maximum(res::KnnResultView) = @inbounds getdist(res, length(res))
@inline Base.minimum(res::KnnResultView) = @inbounds getdist(res, 1)
@inline Base.argmax(res::KnnResultView) = @inbounds getid(res, length(res))
@inline Base.argmin(res::KnnResultView) = @inbounds getid(res, 1)

@inline idview(res::KnnResultView) = @view res.parent.id[eachindex(res), res.i]
@inline distview(res::KnnResultView) = @view res.parent.dist[eachindex(res), res.i]

@inline Base.eachindex(res::KnnResultView) = 1:length(res)
Base.eltype(::KnnResultView) = Pair{Int32,Float32}

##### iterator interface
### KnnResultView

"""
    Base.iterate(res::KnnResultView, state::Int=1)

Support for iteration
"""
function Base.iterate(res::KnnResultView, i::Int=1)
    n = length(res)
    (n == 0 || i > n) && return nothing
    @inbounds res[i], i+1
end