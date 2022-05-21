# This file is a part of SimilaritySearch.jl
struct KnnResultState
    shift::Int
end

export initialstate, KnnResultShift

"""
    KnnResultShift(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResultShift
    id::Vector{Int32}
    dist::Vector{Float32}
    k::Int # number of neighbors
end

function KnnResultShift(k::Integer)
    @assert k > 0
    res = KnnResultShift(Vector{Int32}(undef, 0), Vector{Float32}(undef, 0), k)
    sizehint!(res.id, k)
    sizehint!(res.dist, k)
    res
end

function initialstate(::KnnResultShift)
    KnnResultState(0)
end

function _shifted_fixorder!(res::KnnResultShift, shift)
    sp = shift + 1
    ep = lastindex(res.id)
    id, dist = res.id, res.dist
    @inbounds i, d = id[end], dist[end]
    pos = _find_inspos(dist, sp, ep, d)
    _shift_vector(id, pos, ep, i)
    _shift_vector(dist, pos, ep, d)

    nothing
end

@inline function _find_inspos(dist::Vector, sp, ep, d)
    @inbounds while ep > sp && d < dist[ep-1]
        ep -= 1
    end

    ep
end

@inline function _shift_vector(arr::Vector, sp, ep, val)
    @inbounds while ep > sp
        arr[ep] = arr[ep-1]
        ep -= 1
    end

    arr[ep] = val
end


"""
    push!(res::KnnResultShift, item::Pair)
    push!(res::KnnResultShift, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultShift, st::KnnResultState, id::Integer, dist::Real)
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
    
        _shifted_fixorder!(res, st.shift)
        #_shifted_fixorder!(res.shift, res.id, res.dist)
        return st
    end

    dist >= last(res.dist) && return st

    @inbounds res.id[end], res.dist[end] = id, dist
    _shifted_fixorder!(res, st.shift)
    #_shifted_fixorder!(res.shift, res.id, res.dist)
    st
end

#@inline Base.push!(res::KnnResultShift, st::KnnResultState, id::Integer, dist::Real) = push!(res, st, convert(Int32, id), convert(Float32, dist))
@inline Base.push!(res::KnnResultShift, st::KnnResultState, p::Pair) = push!(res, st, p.first, p.second)

function compact!(res::KnnResultShift, st::KnnResultState, resize_extra)
    shift = st.shift
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
    popfirst!(p::KnnResultShift, st::KnnResultState)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResultShift, st::KnnResultState)
    p = argmin(res, st) => minimum(res, st)
    res.id[1] = 0  # mark as deleted
    p, KnnResultState(st.shift+1)
end

"""
    pop!(res::KnnResultShift, st::KnnResultState)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResultShift, st::KnnResultState)
    pop!(res.id) => pop!(res.dist), st
end

"""
    maxlength(res::KnnResultShift)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultShift) = res.k
@inline Base.length(res::KnnResultShift, st::KnnResultState) = length(res.id) - st.shift

function Base.length(res::KnnResultShift)
    i = 1
    n = length(res.id)
    while i < n && res.id[i] == 0
        i += 1
    end

    n - i + 1
end

"""
    reuse!(res::KnnResultShift)
    reuse!(res::KnnResultShift, k::Integer)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::KnnResultShift, k::Integer=res.k)
    @assert k > 0
    empty!(res.id)
    empty!(res.dist)
    if k > res.k
        sizehint!(res.id, k)
        sizehint!(res.dist, k)
    end
    KnnResultShift(res.id, res.dist, k)
end

"""
    getindex(res::KnnResultShift, st::KnnResultState, i)

Access the i-th item in `res`
"""
@inline function getpair(res::KnnResultShift, st::KnnResultState, i)
    i += st.shift
    @inbounds res.id[i] => res.dist[i]
end

@inline getid(res::KnnResultShift, st::KnnResultState, i) = @inbounds res.id[i+st.shift]
@inline getdist(res::KnnResultShift, st::KnnResultState, i) = @inbounds res.dist[i+st.shift]

@inline Base.last(res::KnnResultShift, st::KnnResultState) = last(res.id) => last(res.dist)
@inline Base.first(res::KnnResultShift, st::KnnResultState) = res.id[st.shift+1] => res.dist[st.shift+1]
@inline Base.maximum(res::KnnResultShift, st::KnnResultState) = last(res.dist)
@inline Base.minimum(res::KnnResultShift, st::KnnResultState) = res.dist[1+st.shift]
@inline Base.argmax(res::KnnResultShift, st::KnnResultState) = last(res.id)
@inline Base.argmin(res::KnnResultShift, st::KnnResultState) = res.id[1+st.shift]

Base.maximum(res::KnnResultShift) = last(res.dist)
Base.argmax(res::KnnResultShift) = last(res.id)
Base.minimum(res::KnnResultShift) = res.dist[_find_start_position(res)]
Base.argmin(res::KnnResultShift) = res.id[_find_start_position(res)]

@inline idview(res::KnnResultShift, st::KnnResultState) = @view res.id[st.shift+1:end]
@inline distview(res::KnnResultShift, st::KnnResultState) = @view res.dist[st.shift+1:end]

@inline Base.eachindex(res::KnnResultShift, st::KnnResultState) = 1:length(res, st)
Base.eltype(res::KnnResultShift) = Pair{Int32,Float32}

##### iterator interface
### KnnResultShift
"""
    Base.iterate(res::KnnResultShift, state::Int=1)

Support for iteration
"""
@inline function _find_start_position(res::KnnResultShift)
    i = 1
    id = res.id
    n = length(id)
    @inbounds while i <= n && id[i] == 0
        i += 1
    end
    
    i
end

function Base.iterate(res::KnnResultShift, i::Int=-1)
    n = length(res.id)
    n == 0 && return nothing
    if i == -1
        i = _find_start_position(res)
    end

    i > n && return nothing
    @inbounds res.id[i] => res.dist[i], i+1
end