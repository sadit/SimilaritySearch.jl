# This file is a part of SimilaritySearch.jl

export maxlength, maxlength, getdist, getid, idview, distview, reuse!
export AbstractKnnResult, KnnResultSet

abstract type AbstractKnnResult end

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
        Vector{Int32}(undef, m)
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
struct KnnResultView <: AbstractKnnResult # <: AbstractVector{Tuple{IdType,DistType}}
    parent::KnnResultSet
    i::Int

    function KnnResultView(parent::KnnResultSet, i::Integer) 
        parent.len[i] = 0
        new(parent, i)
    end
end

function KnnResultView(k::Integer)
    knns = KnnResultSet(k, 1)
    KnnResultView(knns, 1)
end

"""
    reuse!(res::KnnResultView)

Resets `res` to an empty state
"""
@inline function reuse!(res::KnnResultView)
    res.parent.len[res.i] = 0
    res
end

"""
    _shifted_fixorder!(res)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function _shifted_fixorder!(res::KnnResultView)
    k = length(res)
    @inbounds i, d = res[k]
    pos = _find_inspos(res.parent.dist, res.i, 1, k, d)
    _shift_vector(res.parent.id, res.i, pos, k, i)
    _shift_vector(res.parent.dist, res.i, pos, k, d)

    nothing
end

@inline function _find_inspos(dist::Matrix, col, sp, ep, d)
    @inbounds while ep > sp && d < dist[ep-1, col]
        ep -= 1
    end

    ep
end

@inline function _shift_vector(M::Matrix, col, sp, ep, val)
    @inbounds while ep > sp
        M[ep, col] = M[ep-1, col]
        ep -= 1
    end

    M[ep, col] = val
end

### push functions

"""
    push!(res::KnnResultView, item::Pair)
    push!(res::KnnResultView, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultView, id::Integer, dist::Real)
    k = length(res)

    @inbounds if k < maxlength(res)
        k += 1
        res.parent.id[k, res.i] = id
        res.parent.dist[k, res.i] = dist
        res.parent.len[res.i] = k
    
        _shifted_fixorder!(res)
        return true
    end

    dist >= maximum(res) && return false

    @inbounds res.parent.id[k, res.i], res.parent.dist[k, res.i] = id, dist
    _shifted_fixorder!(res)
    true
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
        _popfirst!(res.parent.id, res.i, n) =>  _popfirst!(res.parent.dist, res.i, n)
    end
end

@inline function _popfirst!(M::Matrix, col::Integer, len::Integer)
    @inbounds begin
        s = M[1, col]
        for i in 1:len-1
            M[i, col] = M[i+1, col]
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
        getid(res, n) => getdist(res, n)
    end
end

##### access functions #######

@inline getid(res::KnnResultView, i) = @inbounds res.parent.id[i, res.i] 
@inline getdist(res::KnnResultView, i) = @inbounds res.parent.dist[i, res.i] 

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
@inline Base.length(res::KnnResultView) = @inbounds res.parent.len[res.i]

@inline Base.last(res::KnnResultView) = argmax(res) => maximum(res)
@inline Base.first(res::KnnResultView) = argmin(res) => minimum(res)
@inline Base.maximum(res::KnnResultView) = @inbounds getdist(res, length(res))
@inline Base.minimum(res::KnnResultView) = @inbounds getdist(res, 1)
@inline Base.argmax(res::KnnResultView) = @inbounds getid(res, length(res))
@inline Base.argmin(res::KnnResultView) = @inbounds getid(res, 1)

@inline idview(res::KnnResultView) = @view res.parent.id[1:length(res), res.i]
@inline distview(res::KnnResultView) = @view res.parent.dist[1:length(res), res.i]

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