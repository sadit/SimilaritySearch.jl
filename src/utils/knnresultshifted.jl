# This file is a part of SimilaritySearch.jl

export KnnResultShifted

"""
    KnnResultShifted(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResultShifted <: AbstractKnnResult # <: AbstractVector{Tuple{IdType,DistType}}
    id::Vector{Int32}
    dist::Vector{Float32}
    k::Int  # number of neighbors
end

function KnnResultShifted(k::Integer)
    @assert k > 0
    KnnResultShifted(Vector{Int32}(undef, 0), Vector{Float32}(undef, 0), k)
end

function initialstate(::KnnResultShifted)
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
    push!(res::KnnResultShifted, item::Pair)
    push!(res::KnnResultShifted, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultShifted, st::KnnResultState, id::Integer, dist::Real)
    if length(res, st) < maxlength(res, st)
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

function compact!(res::KnnResultShifted, st::KnnResultState, resize_extra)
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
    popfirst!(p::KnnResultShifted, st::KnnResultState)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResultShifted, st::KnnResultState)
    @inbounds argmin(res, st) => minimum(res, st), KnnResultState(st.pos+1)
end

"""
    pop!(res::KnnResultShifted, st::KnnResultState)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResultShifted, st::KnnResultState)
    pop!(res.id) => pop!(res.dist), st
end

"""
    maxlength(res::KnnResultShifted)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultShifted, st::KnnResultState) = res.k
@inline Base.length(res::KnnResultShifted, st::KnnResultState) = length(res.id) - st.pos

"""
    reuse!(res::KnnResultShifted)
    reuse!(res::KnnResultShifted, k::Integer)

Returns a result set and a new initial state; reuse the memory buffers
"""
@inline function reuse!(res::KnnResultShifted, k::Integer=res.k)
    @assert k > 0
    resize!(res.id, k)
    resize!(res.dist, k)
    res_ = KnnResultShifted(res.id, res.dist, k)
    res_, initialstate(res_)
end

"""
    getindex(res::KnnResultShifted, st::KnnResultState, i)

Access the i-th item in `res`
"""
@inline function Base.get(res::KnnResultShifted, st::KnnResultState, i)
    i += st.pos
    @inbounds res.id[i] => res.dist[i]
end

@inline getid(res::KnnResultShifted, st::KnnResultState, i) = @inbounds res.id[i+st.pos]
@inline getdist(res::KnnResultShifted, st::KnnResultState, i) = @inbounds res.dist[i+st.pos]

@inline Base.last(res::KnnResultShifted, st::KnnResultState) = last(res.id) => last(res.dist)
@inline Base.first(res::KnnResultShifted, st::KnnResultState) = res.id[st.pos+1] => res.dist[st.pos+1]
@inline Base.maximum(res::KnnResultShifted, st::KnnResultState) = last(res.dist)
@inline Base.minimum(res::KnnResultShifted, st::KnnResultState) = res.dist[1+st.pos]
@inline Base.argmax(res::KnnResultShifted, st::KnnResultState) = last(res.id)
@inline Base.argmin(res::KnnResultShifted, st::KnnResultState) = res.id[1+st.pos]
@inline idview(res::KnnResultShifted, st::KnnResultState) = @view res.id[st.pos+1:end]
@inline distview(res::KnnResultShifted, st::KnnResultState) = @view res.dist[st.pos+1:end]
