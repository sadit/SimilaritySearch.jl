# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export KnnResult, maxlength, covrad, maxlength

"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
mutable struct KnnResult{RefType<:Integer,DistType<:Real}
    id::Vector{RefType}
    dist::Vector{DistType}
    k::Int

    function KnnResult(id, dist, k::Integer)
        @assert k > 0
        new{eltype(id), eltype(dist)}(id, dist, Int(k))
    end
end

function KnnResult(k::Integer)
    @assert k > 0
    KnnResult(Int32[], Float32[], k)
end
Base.copy(res::KnnResult) = KnnResult(copy(res.id), copy(res.dist), res.k)

"""
    fixorder!(id, dist)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function fixorder!(id, dist)
    pos = N = length(id)
    @inbounds while pos > 1 && dist[N] < dist[pos-1]
        pos -= 1
    end

    @inbounds if pos < N
        id_, dist_ = last(id), last(dist)
        while N > pos
            id[N], dist[N] = id[N-1], dist[N-1]
            N -= 1
        end

        dist[N] = dist_
        id[N] = id_
    end
end

"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, id::Integer, dist::Real)
    if length(res) < maxlength(res)
        push!(res.id, id)
        push!(res.dist, dist)
        fixorder!(res.id, res.dist)
        return true
    end

    dist >= last(res.dist) && return false
    @inbounds res.id[end], res.dist[end] = id, dist
    fixorder!(res.id, res.dist)
    true
end

@inline Base.push!(res::KnnResult, p::Pair) = push!(res, p.first, p.second)


"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResult)
    popfirst!(res.id), popfirst!(res.dist)
end

"""
    pop!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult)
    pop!(res.id), pop!(res.dist)
end

"""
    length(p::KnnResult)

length returns the number of items in the result set
"""
@inline Base.length(res::KnnResult) = length(res.id)

"""
    maxlength(res::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult) = res.k

"""
    covrad(p::KnnResult)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::KnnResult)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : res.dist[end]

"""
    empty!(res::KnnResult)
    empty!(res::KnnResult, k::Integer)

Clears the content of the result pool. If k is given then the size of the pool is changed; the internal buffer is adjusted
as needed (only grows).
"""
@inline function Base.empty!(res::KnnResult, k::Integer=0)
    empty!(res.id)
    empty!(res.dist)
    if k > res.k 
        sizehint!(res.id, k)
        sizehint!(res.dist, k)
    end

    res.k = max(k, res.k)
end

"""
    getindex(res::KnnResult, i)

Access the i-th item in `res`
"""
@inline Base.getindex(res::KnnResult, i) = res.id[i] => res.dist[i]

"""
    lastindex(res::KnnResult)

Last index of `res`
"""
@inline Base.lastindex(res::KnnResult) = lastindex(res.id)

"""
    eachindex(res::KnnResult)

Iterator of valid item indexes in `res`
"""
@inline Base.eachindex(res::KnnResult) = eachindex(res.id)

##### iterator interface
### KnnResult
"""
    Base.iterate(res::KnnResult, state::Int=1)

Support for iteration
"""
function Base.iterate(res::KnnResult, state::Int=1)
    n = length(res)
    if n == 0 || state > length(res)
        nothing
    else
        @inbounds res[state], state + 1
    end
end

Base.eltype(res::KnnResult{I,F}) where {I,F} = Pair{I,F}