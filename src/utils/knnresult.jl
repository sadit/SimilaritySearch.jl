# This file is a part of SimilaritySearch.jl
using Intersections

export KnnResult, maxlength, covrad, maxlength

"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
mutable struct KnnResult{IdVectorType,DistVectorType}  # <: AbstractVector{Tuple{eltype(IdVectorType),eltype(DistVectorType)}}
    id::IdVectorType
    dist::DistVectorType
    pos::Int
    len::Int
end

function KnnResult(k::Integer=10, F=Float32)
    @assert k > 0
    KnnResult(Vector{Int32}(undef, k), Vector{F}(undef, k), 0, convert(Int, k))
end

function Base.copy(res::KnnResult)
    KnnResult(copy(res.id), copy(res.dist), res.pos, res.len)
end

"""
    fixorder!(res, shift=0)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function fixorder!(res, shift=0)
    sp = shift + 1
    pos = N = lastindex(res)
    id = res.id
    dist = res.dist
    id_, dist_ = last(res)
    
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
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, id::Int32, dist::Float32)
    @inbounds if length(res) < maxlength(res)
        res.pos += 1
        res.id[res.pos] = id
        res.dist[res.pos] = dist
        fixorder!(res)
        return true
    end

    dist >= maximum(res) && return false

    @inbounds res.id[res.pos] = id
    @inbounds res.dist[res.pos] = dist
    fixorder!(res)

    true
end

@inline Base.push!(res::KnnResult, id::Integer, dist::Real) = push!(res, convert(Int32, id), convert(Float32, dist))
@inline Base.push!(res::KnnResult, p::Pair) = push!(res, p.first, p.second)

"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p)) operation
"""
@inline function Base.popfirst!(res::KnnResult)
    p = res.id[1] => res.dist[1]
    @inbounds for i in 1:(res.pos-1)
        res.id[i] = res.id[i+1]
        res.dist[i] = res.dist[i+1]
    end

    res.pos -= 1
    p
end

"""
    pop!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult)
    p = res.id[res.pos] => res.dist[res.pos]
    res.pos -= 1
    p
end

@inline Base.firstindex(res::KnnResult) = 1
@inline Base.lastindex(res::KnnResult) = res.pos
@inline Base.length(res::KnnResult) = res.pos
@inline Base.keys(res::KnnResult) = @view res.id[eachindex(res)]
@inline Base.values(res::KnnResult) = @view res.dist[eachindex(res)]
@inline Base.maximum(res::KnnResult) = @inbounds res.dist[lastindex(res)]
@inline Base.minimum(res::KnnResult) = @inbounds res.dist[firstindex(res)]
@inline Base.argmin(res::KnnResult) = @inbounds res.id[firstindex(res)]
@inline Base.argmax(res::KnnResult) = @inbounds res.id[lastindex(res)]
@inline Base.first(res::KnnResult) = @inbounds res.id[firstindex(res)] => res.dist[firstindex(res)]
@inline Base.last(res::KnnResult) = @inbounds res.id[lastindex(res)] => res.dist[lastindex(res)]
@inline Base.size(res::KnnResult) = (length(res),)

"""
    maxlength(res::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult) = res.len

"""
    covrad(p::KnnResult)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::KnnResult)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res)

"""
    empty!(res::KnnResult)
    empty!(res::KnnResult, k::Integer)

Clears the content of the result pool. If k is given then the size of the pool is changed; the internal buffer is adjusted
as needed (only grows).
"""
@inline function Base.empty!(res::KnnResult, k::Integer=maxlength(res))
    @assert k > 0
    if k > maxlength(res)
        resize!(res.id, k)
        resize!(res.dist, k)
    end
    res.pos = 0
    res.len = k
end

"""
    getindex(res::KnnResult, i)

Access the i-th item in `res`
"""
@inline function Base.getindex(res::KnnResult, i)
    @inbounds res.id[i] => res.dist[i]
end


"""
    eachindex(res::KnnResult)

Iterator of valid item indexes in `res`
"""
@inline Base.eachindex(res::KnnResult) = firstindex(res):lastindex(res)

##### iterator interface
### KnnResult
"""
    Base.iterate(res::KnnResult, state::Int=1)

Support for iteration
"""
function Base.iterate(res::KnnResult, state::Int=1)
    n = length(res)
    if n == 0 || state > n
        nothing
    else
        @inbounds res[state], state + 1
    end
end

Base.eltype(res::KnnResult{I,F}) where {I,F} = Pair{eltype(I),eltype(F)}
#Base.IndexStyle(::Type{<:KnnResult}) = IndexLinear()
