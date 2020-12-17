# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
#=
export KnnResultArray

mutable struct KnnResultArray <: AbstractKnnResult
    k::Int32
    # pool::Vector{Item}
    id::Vector{Int32}
    dist::Vector{Float32}

    function KnnResultArray(k::Integer)
        id = Vector{Int32}()
        dist = Vector{Float32}()
        sizehint!(id, k); sizehint!(dist, k)
        new(k, id, dist)
    end
end

"""
    fix_order!(res::KnnResultArray)

Fixes the sorted state of the array. It implements a kind of insertion sort
It is efficient due to the expected distribution of the items being inserted
(few smaller than the ones already inside)
"""
@inline function fix_order!(K, V)
    n = length(K)
    @inbounds while n > 1
        if K[n] < K[n-1]
            K[n], K[n-1] = K[n-1], K[n]
            V[n], V[n-1] = V[n-1], V[n]
        else
            break
        end
        n -= 1
    end
end

"""
    push!(p::KnnResultArray, item::Item) where T

Appends an item into the result set
"""

@inline function Base.push!(res::KnnResultArray, p::Pair)
    push!(res, p.first, p.second)
end

@inline function Base.push!(res::KnnResultArray, id::Integer, dist::Number)
    n = length(res.id)
    if n < res.k
        # fewer elements than the maximum capacity
        push!(res.id, id)
        push!(res.dist, dist)
        fix_order!(res.dist, res.id)
        return true
    end

    if dist >= farthestdist(res)
        # p.k == length(p.pool) but p.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but p.dist improves the result set
    @inbounds res.id[end] = id
    @inbounds res.dist[end] = dist
    fix_order!(res.dist, res.id)
    true
end

"""
    nearest(p::KnnResultArray)

Return the first item of the result set, the closest item
"""
#@inline nearest(res::KnnResultArray) = Item(first(res.id), first(res.dist))

"""
    farthest(p::KnnResultArray) 

Returns the last item of the result set
"""
#@inline farthest(res::KnnResultArray) = Item(last(res.id), last(res.dist))

"""
    popnearest!(p::KnnResultArray)

Removes and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation
"""
@inline popnearest!(res::KnnResultArray) = Item(popfirst!(res.id), popfirst!(res.dist))

"""
    popfarthest!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline popfarthest!(res::KnnResultArray) = Item(pop!(res.id), pop!(res.dist))

"""
    length(p::KnnResultArray)

length returns the number of items in the result set
"""
@inline Base.length(res::KnnResultArray) = length(res.id)

"""
    maxlength(res::KnnResultArray)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultArray) = res.k

"""
    covrad(p::KnnResultArray)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::KnnResultArray) = length(res.dist) < res.k ? typemax(Float32) : last(res.dist)

@inline nearestid(res::KnnResultArray) = first(res.id)
@inline farthestid(res::KnnResultArray) = last(res.id)

@inline nearestdist(res::KnnResultArray) = first(res.dist)
@inline farthestdist(res::KnnResultArray) = last(res.dist)

"""
    empty!(p::KnnResultArray)

Clears the content of the result pool
"""
@inline function Base.empty!(p::KnnResultArray) 
    empty!(p.id)
    empty!(p.dist)
end

@inline function reset!(p::KnnResultArray, k::Integer)
    empty!(p)
    sizehint!(p.id, k)
    sizehint!(p.dist, k)
    p.k = k
    p
end

##### iterator interface
### KnnResultArray
function Base.iterate(res::KnnResultArray, state::Int=1)
    n = length(res)
    if n == 0 || state > length(res)
        nothing
    else
        @inbounds Item(res.id[state], res.dist[state]), state + 1
    end
end

=#