# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using DataStructures
export KnnResult, maxlength, covrad, maxlength

struct Item
    id::Int32
    dist::Float32
end

Base.isless(a::Item, b::Item) = a.dist < b.dist

"""
    KnnResult(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
mutable struct KnnResult
    pool::BinaryMinMaxHeap{Item}
    k::Int

    function KnnResult(k::Integer)
        @assert k > 0
        h = BinaryMinMaxHeap{Item}()
        sizehint!(h, k)
        new(h, k)
    end
end


Base.copy(res::KnnResult) = KnnResult(res.pool, res.k)

"""
    push!(res::KnnResult, item::Pair)
    push!(res::KnnResult, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResult, id::Integer, dist::Real)
    if length(res) < maxlength(res)
        push!(res.pool, Item(id, dist))
        return true
    end

    dist >= maximum(res.pool).dist && return false
    push!(res.pool, Item(id, dist))
    true
end

@inline Base.push!(res::KnnResult, p::Pair) = push!(res, p.first, p.second)

"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResult)
    popmin!(res.pool)
end

"""
    pop!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResult)
    popmax!(res.pool)
end

"""
    length(p::KnnResult)

length returns the number of items in the result set
"""
@inline Base.length(res::KnnResult) = length(res.pool)

"""
    maxlength(res::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResult) = res.k

"""
    covrad(p::KnnResult)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::KnnResult)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : maximum(res.pool).dist

"""
    empty!(res::KnnResult)
    empty!(res::KnnResult, k::Integer)

Clears the content of the result pool. If k is given then the size of the pool is changed; the internal buffer is adjusted
as needed (only grows).
"""
@inline function Base.empty!(res::KnnResult, k::Integer=res.k)
    @assert k > 0
    empty!(res.pool)
    res.k = k
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

Base.eltype(res::KnnResult) = Item