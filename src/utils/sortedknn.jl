# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Item, KnnResult, SortedKnnResult, maxlength, covrad, nearest, farthest, nearestdist, farthestdist, popnearest!, popfarthest!, reset!


abstract type KnnResult end
struct Item
    id::Int32
    dist::Float32
end

# Base.convert(Item, p::Pair) = Item(p.first, p.second)

mutable struct SortedKnnResult <: KnnResult
    k::Int32
    pool::Vector{Item}

    function SortedKnnResult(k::Integer)
        pool = Vector{Item}()
        sizehint!(pool, k)
        new(k, pool)
    end
end

function KnnResult(k::Integer)
    SortedKnnResult(k)
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
        if K[n].dist < K[n-1].dist
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

    if dist >= farthestdist(res)
        # p.k == length(p.pool) but p.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but p.dist improves the result set
    @inbounds res.pool[end] = Item(id, dist)
    fix_order!(res.pool)
    true
end

"""
    nearest(p::SortedKnnResult)

Return the first item of the result set, the closest item
"""
@inline nearest(res::SortedKnnResult) = first(res.pool)

"""
    farthest(p::SortedKnnResult) 

Returns the last item of the result set
"""
@inline farthest(res::SortedKnnResult) = last(res.pool)

"""
    popnearest!(p::SortedKnnResult)

Removes and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation
"""
@inline popnearest!(res::SortedKnnResult) = popfirst!(res.pool)

"""
    popfarthest!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline popfarthest!(res::SortedKnnResult) = pop!(res.pool)

"""
    length(p::SortedKnnResult)

length returns the number of items in the result set
"""
@inline Base.length(res::SortedKnnResult) = length(res.pool)

"""
    maxlength(res::SortedKnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::SortedKnnResult) = res.k

"""
    covrad(p::SortedKnnResult)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::SortedKnnResult) = length(res.pool) < res.k ? typemax(Float32) : farthestdist(res)

@inline nearestdist(res::SortedKnnResult) = first(res.pool).dist
@inline farthestdist(res::SortedKnnResult) = last(res.pool).dist

"""
    empty!(p::SortedKnnResult)

Clears the content of the result pool
"""
@inline function Base.empty!(p::SortedKnnResult) 
    empty!(p.pool)
end

@inline function reset!(p::SortedKnnResult, k::Integer)
    empty!(p)
    sizehint!(p.pool, k)
    p.k = k
    p
end

##### iterator interface
### SortedKnnResult
function Base.iterate(res::SortedKnnResult, state::Int=1)
    n = length(res)
    if n == 0 || state > length(res)
        nothing
    else
        @inbounds res.pool[state], state + 1
    end
end
