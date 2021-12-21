# This file is a part of SimilaritySearch.jl
using Intersections

export AbstractKnnResult, KnnResultState, maxlength, maxlength, getpair, getdist, getid, initialstate, idview, distview, reuse!

struct KnnResultState
    pos::Int
end

abstract type AbstractKnnResult end

export KnnResult

"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResult <: AbstractKnnResult # <: AbstractVector{Tuple{IdType,DistType}}
    id::Vector{Int32}
    dist::Vector{Float32}
    k::Int  # number of neighbors
end

function KnnResult(k::Integer)
    @assert k > 0
    res = KnnResult(Vector{Int32}(undef, 0), Vector{Float32}(undef, 0), k)
    sizehint!(res.id, k)
    sizehint!(res.dist, k)
    res
end

function initialstate(::KnnResult)
    KnnResultState(0)
end

"""
    _shifted_fixorder!(res, shift=0)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function _shifted_fixorder!(res, st::KnnResultState)
    sp = st.pos + 1
    pos = N = lastindex(res.id)
    id = res.id
    dist = res.dist
    id_, dist_ = res.id[end], res.dist[end]
    
    #pos = doublingsearch(dist, dist_, sp, N)
    #pos = binarysearch(dist, dist_, sp, N)
    #if N > 16
    #    pos = doublingsearchrev(dist, dist_, sp, N)::Int
    #else
        @inbounds while pos > sp && dist_ < dist[pos-1]
            pos -= 1
        end
    #end

    @inbounds if pos < N
        while N > pos
            id[N] = id[N-1]
            dist[N] = dist[N-1]
            N -= 1
        end

        dist[N] = dist_
        id[N] = id_
    end

    nothing
end


"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, st::KnnResultState, id::Integer, dist::Real)
    if length(res, st) < maxlength(res)
        k = res.k
        if length(res.id) >= 2k-1
            compact!(res, st, 1)
            st = KnnResultState(0)
            @inbounds res.id[end], res.dist[end] = id, dist
        else
            push!(res.id, id)
            push!(res.dist, dist)
        end
    
        _shifted_fixorder!(res, st)
        #_shifted_fixorder!(res.shift, res.id, res.dist)
        return st
    end

    dist >= last(res.dist) && return st

    @inbounds res.id[end], res.dist[end] = id, dist
    _shifted_fixorder!(res, st)
    #_shifted_fixorder!(res.shift, res.id, res.dist)
    st
end

@inline Base.push!(res::AbstractKnnResult, st::KnnResultState, id::Integer, dist::Real) = push!(res, st, convert(Int32, id), convert(Float32, dist))
@inline Base.push!(res::AbstractKnnResult, st::KnnResultState, p::Pair) = push!(res, st, p.first, p.second)


function compact!(res::KnnResult, st::KnnResultState, resize_extra)
    shift = st.pos
    if shift > 0
        n = length(res, st)
        j = shift
        @inbounds for i in 1:n
            j += 1
            res.id[i] = res.id[j]
            res.dist[i] = res.dist[j]
        end

        resize!(res.id, n+resize_extra)
        resize!(res.dist, n+resize_extra)
    end

    res
end

"""
    popfirst!(p::KnnResult, st::KnnResultState)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResult, st::KnnResultState)
    p = argmin(res, st) => minimum(res, st)
    res.id[1] = 0  # mark as deleted
    p, KnnResultState(st.pos+1)
end

"""
    pop!(res::KnnResult, st::KnnResultState)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult, st::KnnResultState)
    pop!(res.id) => pop!(res.dist), st
end

"""
    maxlength(res::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult) = res.k
@inline Base.length(res::KnnResult, st::KnnResultState) = length(res.id) - st.pos

function Base.length(res::KnnResult)
    i = 1
    n = length(res.id)
    while i < n && res.id[i] == 0
        i += 1
    end

    n - i + 1
end

"""
    reuse!(res::KnnResult)
    reuse!(res::KnnResult, k::Integer)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::KnnResult, k::Integer=res.k)
    @assert k > 0
    empty!(res.id)
    empty!(res.dist)
    if k > res.k
        sizehint!(res.id, k)
        sizehint!(res.dist, k)
    end
    KnnResult(res.id, res.dist, k)
end

"""
    getindex(res::KnnResult, st::KnnResultState, i)

Access the i-th item in `res`
"""
@inline function getpair(res::KnnResult, st::KnnResultState, i)
    i += st.pos
    @inbounds res.id[i] => res.dist[i]
end

@inline getid(res::KnnResult, st::KnnResultState, i) = @inbounds res.id[i+st.pos]
@inline getdist(res::KnnResult, st::KnnResultState, i) = @inbounds res.dist[i+st.pos]

@inline Base.last(res::KnnResult, st::KnnResultState) = last(res.id) => last(res.dist)
@inline Base.first(res::KnnResult, st::KnnResultState) = res.id[st.pos+1] => res.dist[st.pos+1]
@inline Base.maximum(res::KnnResult, st::KnnResultState) = last(res.dist)
@inline Base.minimum(res::KnnResult, st::KnnResultState) = res.dist[1+st.pos]
@inline Base.argmax(res::KnnResult, st::KnnResultState) = last(res.id)
@inline Base.argmin(res::KnnResult, st::KnnResultState) = res.id[1+st.pos]

Base.maximum(res::KnnResult) = last(res.dist)
Base.argmax(res::KnnResult) = last(res.id)
Base.minimum(res::KnnResult) = res.dist[_find_start_position(res)]
Base.argmin(res::KnnResult) = res.id[_find_start_position(res)]

@inline idview(res::KnnResult, st::KnnResultState) = @view res.id[st.pos+1:end]
@inline distview(res::KnnResult, st::KnnResultState) = @view res.dist[st.pos+1:end]

@inline Base.eachindex(res::AbstractKnnResult, st::KnnResultState) = 1:length(res, st)
Base.eltype(res::AbstractKnnResult) = Pair{Int32,Float32}

##### iterator interface
### KnnResult
"""
    Base.iterate(res::AbstractKnnResult, state::Int=1)

Support for iteration
"""
@inline function _find_start_position(res::KnnResult)
    i = 1
    id = res.id
    n = length(id)
    @inbounds while i <= n && id[i] == 0
        i += 1
    end
    
    i
end

function Base.iterate(res::KnnResult, i::Int=-1)
    n = length(res.id)
    n == 0 && return nothing
    if i == -1
        i = _find_start_position(res)
    end

    i > n && return nothing
    @inbounds res.id[i] => res.dist[i], i+1
end