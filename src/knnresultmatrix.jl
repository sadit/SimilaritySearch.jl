# This file is a part of SimilaritySearch.jl
using Intersections

export KnnResultMatrix, KnnResultState, maxlength, maxlength, getpair, getdist, getid, initialstate, idview, distview, reuse!

struct KnnResultState
    pos::Int
end

abstract type AbstractKnnResult end
"""
    KnnResultMatrix(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
struct KnnResultMatrix <: AbstractKnnResult  # <: AbstractVector{Tuple{eltype(IdVectorType),eltype(DistVectorType)}}
    id::Matrix{Int32}
    dist::Matrix{Float32}
    col::Int
end

function KnnResultMatrix(k::Integer=10)
    @assert k > 0
    KnnResultMatrix(Matrix{Int32}(undef, k, 1), Matrix{Float32}(undef, k, 1), 1)
end

function KnnResultMatrix(id::Matrix{Int32}, dist::Matrix{Float32}, col::Integer)
    KnnResultMatrix(id, dist, col)
end

function initialstate(::KnnResultMatrix)
    KnnResultState(0)
end


"""
    fixorder!(res::KnnResultMatrix, shift=0)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function fixorder!(res::KnnResultMatrix, st::KnnResultState)
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
    push!(res::KnnResultMatrix, item::Pair)
    push!(res::KnnResultMatrix, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultMatrix, st::KnnResultState, id::Int32, dist::Float32)
    pos = st.pos
    @inbounds if pos < maxlength(res)
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
    popfirst!(p::KnnResultMatrix)

Removes and returns the nearest neeighbor pair from the pool, an O(length(p)) operation
"""
@inline function Base.popfirst!(res::KnnResultMatrix, st::KnnResultState)
    p = res.id[1, res.col] => res.dist[1, res.col]
    pos = st.pos - 1
    @inbounds for i in 1:pos
        res.id[i, res.col] = res.id[i+1, res.col]
        res.dist[i, res.col] = res.dist[i+1, res.col]
    end
    res.id[st.pos, res.col] = 0
    p, KnnResultState(pos)
end

"""
    pop!(p, st::KnnResultState)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResultMatrix, st::KnnResultState)
    p = res.id[st.pos, res.col] => res.dist[st.pos, res.col]
    res.id[st.pos, res.col] = 0  # zero is special (marks final element)
    p, KnnResultState(st.pos - 1)
end

"""
    maxlength(res::KnnResultMatrix)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultMatrix) = size(res.id, 1)
@inline Base.length(res::KnnResultMatrix, st::KnnResultState) = st.pos
@inline function Base.length(res::KnnResultMatrix)
    i = 1
    n = maxlength(res)
    @inbounds while i <= n && res.id[i, res.col] != 0
        i += 1
    end

    i - 1
end

"""
    getindex(res::KnnResultMatrix, st::KnnResultState, i)

Access the i-th item in `res`
"""
@inline function getpair(res::KnnResultMatrix, st::KnnResultState, i)
    @inbounds res.id[i, res.col] => res.dist[i, res.col]
end

@inline getid(res::KnnResultMatrix, st::KnnResultState, i) = @inbounds res.id[i, res.col]
@inline getdist(res::KnnResultMatrix, st::KnnResultState, i) = @inbounds res.dist[i, res.col]

@inline Base.last(res::KnnResultMatrix, st::KnnResultState) = getpair(res, st, st.pos)
@inline Base.first(res::KnnResultMatrix, st::KnnResultState) = getpair(res, st, 1)
@inline Base.maximum(res::KnnResultMatrix, st::KnnResultState) = getdist(res, st, st.pos)
@inline Base.minimum(res::KnnResultMatrix, st::KnnResultState) = getdist(res, st, 1)
@inline Base.argmax(res::KnnResultMatrix, st::KnnResultState) = getid(res, st, st.pos)
@inline Base.argmin(res::KnnResultMatrix, st::KnnResultState) = getid(res, st, 1)
@inline idview(res::KnnResultMatrix, st::KnnResultState) = @view res.id[1:st.pos, res.col]
@inline distview(res::KnnResultMatrix, st::KnnResultState) = @view res.dist[1:st.pos, res.col]

@inline Base.eachindex(res::AbstractKnnResult, st::KnnResultState) = 1:length(res, st)

function Base.iterate(res::KnnResultMatrix, i::Int=1)
    n = maxlength(res)
    (n == 0 || i > n) && return nothing
    @inbounds id = res.id[i, res.col]
    if id == 0
        nothing
    else
        @inbounds dist = res.dist[i, res.col]
        id => dist, i + 1
    end
end

Base.eltype(res::AbstractKnnResult) = Pair{Int32,Float32}