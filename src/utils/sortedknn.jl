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
        v = Vector{Item}()
        sizehint!(v, k)
        new(k, v)
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
@inline function fix_order!(arr)
    n = length(arr)
    @inbounds while n > 1
        if arr[n].dist < arr[n-1].dist
            arr[n], arr[n-1] = arr[n-1], arr[n]
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

    if dist >= farthest(res).dist
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
@inline nearest(p::SortedKnnResult) = first(p.pool)

"""
    farthest(p::SortedKnnResult) 

Returns the last item of the result set
"""
@inline farthest(p::SortedKnnResult) = last(p.pool)

"""
    popnearest!(p::SortedKnnResult)

Removes and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation
"""
@inline popnearest!(p::SortedKnnResult) = popfirst!(p.pool)

"""
    popfarthest!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline popfarthest!(p::SortedKnnResult) = pop!(p.pool)

"""
    length(p::SortedKnnResult)

length returns the number of items in the result set
"""
@inline Base.length(p::SortedKnnResult) = length(p.pool)

"""
    maxlength(p::SortedKnnResult)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(p::SortedKnnResult) = p.k

"""
    covrad(p::SortedKnnResult)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(p::SortedKnnResult) = length(p.pool) < p.k ? typemax(Float32) : last(p.pool).dist

@inline nearestdist(p::SortedKnnResult) = first(p.pool).dist
@inline farthestdist(p::SortedKnnResult) = last(p.pool).dist

"""
    empty!(p::SortedKnnResult)

Clears the content of the result pool
"""
@inline Base.empty!(p::SortedKnnResult) = empty!(p.pool)

@inline function reset!(p::SortedKnnResult, k::Integer)
    empty!(p)
    sizehint!(p.pool, p.k)
    p.k = k
    p
end

##### iterator interface
### SortedKnnResult
function Base.iterate(p::SortedKnnResult)
    return length(p.pool) == 0 ? nothing : (p.pool[1], 2)
end

function Base.iterate(p::SortedKnnResult, state::Int)
    if state > length(p)
        return nothing
    end

    @inbounds return p.pool[state], state + 1
end
