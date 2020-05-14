# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
using DataStructures: heappush!, heappop!, top, percolate_down!
using Base.Order: Reverse
export KnnResultHeap

# Base.convert(Item, p::Pair) = Item(p.first, p.second)

mutable struct KnnResultHeap <: KnnResult
    k::Int32
    heap::Vector{Item}
    minimum::Item

    function KnnResultHeap(k::Integer)
        p = Vector{Item}()
        sizehint!(p, k)
        new(k, p, Item(0, typemax(Float32)))
    end
end


function KnnResult(k::Integer)
    KnnResultArray(k)
end

"""
    push!(p::KnnResultHeap, item::Item) where T

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultHeap, p::Pair)
    push!(res, p.first, p.second)
end

@inline function Base.push!(res::KnnResultHeap, id::Integer, dist::Number)
    n = length(res.heap)
    if n < res.k
        # fewer elements than the maximum capacity
        p = Item(id, dist)
        if p < res.minimum
            res.minimum = p
        end
        heappush!(res.heap, p, Base.Order.Reverse)
        return true
    end

    if dist >= farthestdist(res)
        # p.k == length(p.pool) but p.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but p.dist improves the result set
    # res.heap[1] = Item(id, dist)
    p = Item(id, dist)
    if p < res.minimum
        res.minimum = p
    end

    percolate_down!(res.heap, 1, p, Base.Order.Reverse)
    true
end

"""
    nearest(p::KnnResultHeap)

Return the first item of the result set, the closest item
"""
#@inline nearest(res::KnnResultHeap) = Item(first(res.id), first(res.dist))

"""
    farthest(p::KnnResultHeap) 

Returns the last item of the result set
"""
#@inline farthest(res::KnnResultHeap) = Item(last(res.id), last(res.dist))

"""
    popnearest!(p::KnnResultHeap)

Removes and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation
"""
# @inline popnearest!(res::KnnResultHeap) = popmin!(res.pool)

"""
    popfarthest!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
# @inline popfarthest!(res::KnnResultHeap) = popmax!(res.pool)
#=
The _hsort! function is based on heap sort implementation of SortingAlgorithms.jl

The SortingAlgorithms.jl package is licensed under the MIT Expat License:

> Copyright (c) 2013-2014: Kevin Squire, Stefan Karpinski, Jeff Bezanson.
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#
function _hsort!(v::Vector{Item})
    for i = length(v):-1:2
        # Swap the root with i, the last unsorted position
        x = v[i]
        v[i] = v[1]
        # The heap portion now ends at position i-1, but needs fixing up
        # starting with the root
        percolate_down!(v, 1, x, Base.Order.Reverse, i-1)
    end

    v
end

@inline sortresults!(res::KnnResultHeap) = _hsort!(res.heap)

"""
    length(p::KnnResultHeap)

length returns the number of items in the result set
"""
@inline Base.length(res::KnnResultHeap) = length(res.heap)

"""
    maxlength(res::KnnResultHeap)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultHeap) = res.k

"""
    covrad(p::KnnResultHeap)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::KnnResultHeap) = length(res.heap) < res.k ? typemax(Float32) : res.heap[1].dist

@inline nearestid(res::KnnResultHeap) = res.minimum.id
@inline farthestid(res::KnnResultHeap) = res.heap[1].id

@inline nearestdist(res::KnnResultHeap) = res.minimum.dist
@inline farthestdist(res::KnnResultHeap) = res.heap[1].dist

"""
    empty!(p::KnnResultHeap)

Clears the content of the result pool
"""
@inline function Base.empty!(p::KnnResultHeap) 
    empty!(p.heap)
end

@inline function reset!(p::KnnResultHeap, k::Integer)
    empty!(p.heap)
    sizehint!(p.heap, k)
    p.k = k
    p
end

##### iterator interface
### KnnResultHeap
Base.iterate(res::KnnResultHeap) = iterate(res.heap)
Base.iterate(res::KnnResultHeap, s) = iterate(res.heap, s)
