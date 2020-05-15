# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Item, AbstractKnnResult, KnnResult, maxlength, covrad, nearestid, farthestid, nearestdist, farthestdist, sortresults!, reset!, popnearest!

abstract type AbstractKnnResult end

struct Item
    id::Int32
    dist::Float32
end

Base.isless(a::Item, b::Item) = isless(a.dist, b.dist)

mutable struct KnnResult <: AbstractKnnResult
    k::Int32
    pool::Vector{Item}

    function KnnResult(k::Integer)
        pool = Item[]
        sizehint!(pool, k)
        new(k, pool)
    end
end

"""
    fix_order!(res::KnnResult)

Fixes the sorted state of the array. It implements a kind of insertion sort
It is efficient due to the expected distribution of the items being inserted
(few smaller than the ones already inside)
"""
@inline function fix_order!(K)
    n = length(K)
    @inbounds while n > 1
        if K[n] < K[n-1]
            K[n], K[n-1] = K[n-1], K[n]
        else
            break
        end
        n -= 1
    end
end

"""
    push!(p::KnnResult, item::Item) where T

Appends an item into the result set
"""

@inline function Base.push!(res::KnnResult, p::Pair)
    push!(res, p.first, p.second)
end

@inline function Base.push!(res::KnnResult, id::Integer, dist::Number)
    n = length(res.pool)
    if n < res.k
        # fewer elements than the maximum capacity
        push!(res.pool, Item(id, dist))
        fix_order!(res.pool)
        return true
    end

    if dist >= res.pool[end].dist
        # p.k == length(p.pool) but p.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but p.dist improves the result set
    @inbounds res.pool[end] = Item(id, dist)
    fix_order!(res.pool)
    true
end

"""
    nearest(p::KnnResult)

Return the first item of the result set, the closest item
"""
@inline nearest(res::KnnResult) = @inbounds res.pool[1]

"""
    farthest(p::KnnResult) 

Returns the last item of the result set
"""
@inline farthest(res::KnnResult) = @inbounds res.pool[end]

"""
    popnearest!(p::KnnResult)

Removes and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation
"""
@inline popnearest!(res::KnnResult) = popfirst!(res.pool)

"""
    popfarthest!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline popfarthest!(res::KnnResult) = pop!(res.pool)

sortresults!(res::KnnResult) = res.pool

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
@inline covrad(res::KnnResult) = length(res.pool) < res.k ? typemax(Float32) : res.pool[end].dist

@inline nearestid(res::KnnResult) = nearest(res).id
@inline farthestid(res::KnnResult) = farthest(res).id

@inline nearestdist(res::KnnResult) = nearest(res).dist
@inline farthestdist(res::KnnResult) = farthest(res).dist

"""
    empty!(p::KnnResult)

Clears the content of the result pool
"""
@inline function Base.empty!(p::KnnResult) 
    empty!(p.pool)
end

@inline function reset!(p::KnnResult, k::Integer)
    empty!(p)
    sizehint!(p.pool, k)
    p.k = k
    p
end

##### iterator interface
### KnnResult
function Base.iterate(res::KnnResult, state::Int=1)
    n = length(res)
    if n == 0 || state > length(res)
        nothing
    else
        @inbounds res.pool[state], state + 1
    end
end
