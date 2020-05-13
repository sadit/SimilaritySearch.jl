# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using DataStructures
export HeapKnnResult

mutable struct HeapKnnResult <: KnnResult
    k::Int32
    covrad::Float32
    pool::BinaryMinHeap{Item}
    function HeapKnnResult(k::Integer)
        pool = BinaryMinHeap{Item}()
        sizehint!(pool, k)
        new(k, typemax(Float32), pool)
    end
end


"""
    push!(p::SortedKnnResult, objID::Integer, dist::Number)

Appends an item into the result set
"""
@inline function Base.push!(p::SortedKnnResult, objID::Integer, dist::Number)
    n = length(p.pool)
    if n < p.k
        # fewer items than the maximum capacity
        push!(p.pool, Item(objID, dist))
        fix_order!(p.pool)
        return true
    end

    if dist >= last(p).dist
        # p.k == length(p.pool) but item.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but item.dist improves the result set
    @inbounds p.pool[end] = Item(objID, dist)
    fix_order!(p.pool)
    true
end

