# This file is a part of SimilaritySearch.jl
using Intersections

export KnnResult, KnnResultState, maxlength, maxlength, getpair, getdist, getid, initialstate, idview, distview

struct KnnResultState
    pos::Int
end

abstract type AbstractKnnResult end
"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResult <: AbstractKnnResult  # <: AbstractVector{Tuple{eltype(IdVectorType),eltype(DistVectorType)}}
    id::Matrix{Int32}
    dist::Matrix{Float32}
    col::Int
end

function KnnResult(k::Integer=10)
    @assert k > 0
    KnnResult(Matrix{Int32}(undef, k, 1), Matrix{Float32}(undef, k, 1), 1)
end

function KnnResult(id::Matrix{Int32}, dist::Matrix{Float32}, col::Integer)
    KnnResult(id, dist, col)
end

function initialstate(::KnnResult)
    KnnResultState(0)
end


"""
    fixorder!(res::KnnResult, shift=0)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function fixorder!(res::KnnResult, st::KnnResultState)
    sp = 1
    pos = N = st.pos
    id = res.id
    dist = res.dist
    col = res.col
    id_, dist_ = id[pos, col], dist[pos, col]
    
    #pos = doublingsearch(dist, dist_, sp, N)
    #pos = binarysearch(dist, dist_, sp, N)
    #=if N > 64
        pos = doublingsearchrev(dist, dist_, sp, N)::Int
    else=#
        @inbounds while pos > sp && dist_ < dist[pos-1, col]
            pos -= 1
        end
    #end

    @inbounds if pos < N
        while N > pos
            id[N, col] = id[N-1, col]
            dist[N, col] = dist[N-1, col]
            N -= 1
        end

        dist[N, col] = dist_
        id[N, col] = id_
    end

    nothing
end

"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, st::KnnResultState, id::Int32, dist::Float32)
    pos = st.pos
    @inbounds if pos < maxlength(res, st)
        pos += 1
        res.id[pos, res.col] = id
        res.dist[pos, res.col] = dist
        st = KnnResultState(pos)
        fixorder!(res, st)
        return st
    end

    @inbounds dist >= getdist(res, st, pos) && return st

    @inbounds res.id[pos, res.col] = id
    @inbounds res.dist[pos, res.col] = dist
    fixorder!(res, st)

    st
end

@inline Base.push!(res::AbstractKnnResult, st::KnnResultState, id::Integer, dist::Real) = push!(res, st, convert(Int32, id), convert(Float32, dist))
@inline Base.push!(res::AbstractKnnResult, st::KnnResultState, p::Pair) = push!(res, st, p.first, p.second)

"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighbor pair from the pool, an O(length(p)) operation
"""
@inline function Base.popfirst!(res::KnnResult, st::KnnResultState)
    p = res.id[1, res.col] => res.dist[1, res.col]
    pos = st.pos - 1
    @inbounds for i in 1:pos
        res.id[i, res.col] = res.id[i+1, res.col]
        res.dist[i, res.col] = res.dist[i+1, res.col]
    end

    p, KnnResultState(pos)
end

"""
    pop!(p, st::KnnResultState)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult, st::KnnResultState)
    res.id[st.pos, res.col] => res.dist[st.pos, res.col], KnnResultState(st.pos - 1)
end

"""
    maxlength(res::KnnResult, st::KnnResultState)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult, st::KnnResultState) = size(res.id, 1)
@inline Base.length(res::KnnResult, st::KnnResultState) = st.pos

"""
    getindex(res::KnnResult, st::KnnResultState, i)

Access the i-th item in `res`
"""
@inline function getpair(res::KnnResult, st::KnnResultState, i)
    @inbounds res.id[i, res.col] => res.dist[i, res.col]
end

@inline getid(res::KnnResult, st::KnnResultState, i) = @inbounds res.id[i, res.col]
@inline getdist(res::KnnResult, st::KnnResultState, i) = @inbounds res.dist[i, res.col]

@inline Base.last(res::KnnResult, st::KnnResultState) = getpair(res, st, st.pos)
@inline Base.first(res::KnnResult, st::KnnResultState) = getpair(res, st, 1)
@inline Base.maximum(res::KnnResult, st::KnnResultState) = getdist(res, st, st.pos)
@inline Base.minimum(res::KnnResult, st::KnnResultState) = getdist(res, st, 1)
@inline Base.argmax(res::KnnResult, st::KnnResultState) = getid(res, st, st.pos)
@inline Base.argmin(res::KnnResult, st::KnnResultState) = getid(res, st, 1)
@inline idview(res::KnnResult, st::KnnResultState) = @view res.id[1:st.pos, res.col]
@inline distview(res::KnnResult, st::KnnResultState) = @view res.dist[1:st.pos, res.col]

@inline Base.eachindex(res::AbstractKnnResult, st::KnnResultState) = 1:length(res, st)
