# This file is a part of SimilaritySearch.jl

export KnnResultVector, reuse!
"""
    KnnResultVector(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResultVector <: AbstractKnnResult  # <: AbstractVector{Tuple{eltype(IdVectorType),eltype(DistVectorType)}}
    id::Vector{Int32}
    dist::Vector{Float32}
end

function KnnResultVector(k::Integer=10)
    @assert k > 0
    KnnResultVector(Vector{Int32}(undef, k), Vector{Float32}(undef, k))
end

"""
    fixorder!(res, shift=0)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function fixorder!(res::KnnResultVector, N, shift=0)
    sp = shift + 1
    pos = N
    id = res.id
    dist = res.dist
    id_, dist_ = res.id[N], res.dist[N]
    
    #pos = doublingsearch(dist, dist_, sp, N)
    #pos = binarysearch(dist, dist_, sp, N)
    if N > 16
        pos = doublingsearchrev(dist, dist_, sp, N)::Int
    else
        @inbounds while pos > sp && dist_ < dist[pos-1]
            pos -= 1
        end
    end

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
    push!(res::KnnResultVector, pos, item::Pair)
    push!(res::KnnResultVector, pos, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultVector, pos, id::Int32, dist::Float32)
    @inbounds if pos < maxlength(res)
        pos += 1
        res.id[pos] = id
        res.dist[pos] = dist
        fixorder!(res, pos)
        pos
    end

    @inbounds dist >= getdist(res, pos) && return pos
    @inbounds res.id[pos] = id
    @inbounds res.dist[pos] = dist
    fixorder!(res, pos)

    pos
end

"""
    popfirst!(p::KnnResultVector, pos)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p)) operation
"""
@inline function Base.popfirst!(res::KnnResultVector, pos)
    p = res.id[1] => res.dist[1]
    pos -= 1
    @inbounds for i in 1:pos
        res.id[i] = res.id[i+1]
        res.dist[i] = res.dist[i+1]
    end

    p, pos
end

"""
    pop!(p, pos)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResultVector, pos)
    p = res.id[pos] => res.dist[pos]
    p, pos-1
end

"""
    maxlength(res::KnnResultVector)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultVector) = length(res.id)


"""
    reuse!(res::KnnResultVector, k::Integer)

Clears the content of the result pool. If k is given then the size of the pool is changed.
"""
@inline function reuse!(res::KnnResultVector, k)
    if k != maxlength(res)
        resize!(res.id, k)
        resize!(res.dist, k)
    end

    res
end

"""
    getindex(res::KnnResultVector, i)

Access the i-th item in `res`
"""
@inline function getpair(res::KnnResultVector, st::KnnResultState, i)
    @inbounds getid(res, st, i) => getdist(res, st, i)
end

@inline getid(res::KnnResultVector, st::KnnResultState, i) = @inbounds res.id[i+st.pos]
@inline getdist(res::KnnResultVector, st::KnnResultState, i) = @inbounds res.dist[i+st.pos]

@inline Base.last(res::KnnResultVector, st::KnnResultState) = last(res.id) => last(res.dist)
@inline Base.first(res::KnnResultVector, st::KnnResultState) = res.id[st.pos] => res.dist[st.pos]
@inline Base.maximum(res::KnnResultVector, st::KnnResultState) = last(res.dist)
@inline Base.minimum(res::KnnResultVector, st::KnnResultState) = res.dist[st.pos]
@inline Base.argmax(res::KnnResultVector, st::KnnResultState) = last(res.id)
@inline Base.argmin(res::KnnResultVector, st::KnnResultState) = res.id[st.pos]
@inline idview(res::KnnResultVector, st::KnnResultState) = @view res.id[st.pos:end]
@inline distview(res::KnnResultVector, st::KnnResultState) = @view res.dist[st.pos:end]
