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
    k::Int32  # number of neighbors
    shift::Int32 # shift position of the first element (to support popfirst! efficiently)
    
    function KnnResult(id, dist, k::Integer)
        @assert k > 0
        new{eltype(id), eltype(dist)}(id, dist, Int32(k), 0)
    end
end

KnnResult(k::Integer) = KnnResult(Int32[], Float32[], k)

Base.copy(res::KnnResult) = KnnResult(copy(res.id), copy(res.dist), res.k)

"""
    fixorder!(sp, id, dist)

Sorts the result in place; the possible element out of order is on the last entry always.
It implements a kind of insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected just a few elements smaller than the current ones)
"""
function fixorder!(shift, id, dist)
    sp = shift + 1
    pos = N = length(id)
    id_, dist_ = last(id), last(dist)
    #if N-sp > 15
    #    D = @view dist[sp:end-1]
    #    pos = Base.Sort.searchsortedlast(D, dist_) + sp
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
end

"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, id::Integer, dist::Real)
    n = length(res)
    if n < maxlength(res)
        k = res.k
        if length(res.id) >= 2k-1
            j = res.shift
            @inbounds for i in 1:n
                j += 1
                res.id[i] = res.id[j]
                res.dist[i] = res.dist[j]
            end

            resize!(res.id, n+1)
            resize!(res.dist, n+1)
            res.shift = 0

            @inbounds res.id[end], res.dist[end] = id, dist
        else
            push!(res.id, id)
            push!(res.dist, dist)
        end
        fixorder!(res.shift, res.id, res.dist)
        return true
    end

    dist >= last(res.dist) && return false
    @inbounds res.id[end], res.dist[end] = id, dist
    fixorder!(res.shift, res.id, res.dist)
    true
end

@inline Base.push!(res::KnnResult, p::Pair) = push!(res, p.first, p.second)


"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResult)
    res.shift += 1
    @inbounds res.id[res.shift] => res.dist[res.shift]
    #popfirst!(res.id), popfirst!(res.dist)
end

"""
    pop!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult)
    pop!(res.id) => pop!(res.dist)
end

@inline Base.keys(res::KnnResult) = @view res.id[eachindex(res)]
@inline Base.values(res::KnnResult) = @view res.dist[eachindex(res)]
@inline Base.maximum(res::KnnResult) = last(res.dist)
@inline Base.minimum(res::KnnResult) = @inbounds res.dist[firstindex(res)]
@inline Base.firstindex(res::KnnResult) = 1+res.shift
@inline Base.lastindex(res::KnnResult) = lastindex(res.id)
@inline Base.first(res::KnnResult) = @inbounds res.id[firstindex(res)] => res.dist[firstindex(res)]
@inline Base.last(res::KnnResult) = @inbounds res.id[lastindex(res)] => res.dist[lastindex(res)]
@inline Base.argmin(res::KnnResult) = @inbounds res.id[firstindex(res)]
@inline Base.argmax(res::KnnResult) = @inbounds res.id[lastindex(res)]
@inline Base.length(res::KnnResult) = length(res.id) - res.shift

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
@inline function Base.empty!(res::KnnResult, k::Integer=res.k)
    @assert k > 0
    empty!(res.id)
    empty!(res.dist)
    res.k = k
    res.shift = 0
end

"""
    getindex(res::KnnResult, i)

Access the i-th item in `res`
"""
@inline function Base.getindex(res::KnnResult, i)
    i += res.shift
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

Base.eltype(res::KnnResult{I,F}) where {I,F} = Pair{I,F}