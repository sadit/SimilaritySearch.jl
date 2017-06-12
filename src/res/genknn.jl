#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http:#www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import Base: push!, shift!, pop!, length, start, done, next, eltype, last, first, clear!
export Item, KnnResult, maxlength, covrad, SlugKnnResult, NnResult

struct Item{T}
    objID::T
    dist::Float64
end

mutable struct KnnResult{T} <: Result
    k::Int
    pool::Vector{Item{T}}
end

function KnnResult(T::Type, k::Int)
    v = Vector{Item{T}}()
    sizehint!(v, k)
    KnnResult(k, v)
end

KnnResult(k::Integer) = KnnResult(Int64, k)
SlugKnnResult(k::Integer) = KnnResult(Int64, k)
NnResult() = KnnResult(Int64, 1)


"""
fix_order! fixes the sorted state of the array. It implements a kind of insertion sort
It is efficient due to the expected distribution of the items being inserted
(few smaller than the ones already inside)
"""
@inline function fix_order!(res::KnnResult{T}) where T
    arr::Vector{Item{T}} = res.pool

    item = arr[end]
    i = length(arr)
    @inbounds while i > 1
        if item.dist < arr[i-1].dist
            arr[i] = arr[i-1]
        else
            arr[i] = item
            return
        end
        i -= 1
    end
    arr[1] = item
end

"""
push! appends an item to the end of the result set
"""

push!(p::KnnResult{Int64}, objID::I, dist::F) where {I <: Union{Int32,Int16}, F <: Real} = push!(p, convert(Int64, objID), convert(Float64, dist))
push!(p::KnnResult{T}, objID::T, dist::F) where {T, F <: Union{Float16, Float32}} = push!(p, objID, convert(Float64, dist))

function push!(p::KnnResult{T}, objID::T, dist::Float64) where T
    if length(p.pool) < p.k
        # fewer items than the maximum capacity
        push!(p.pool, Item(objID, dist))
        fix_order!(p)
        return true
    end

    @inbounds last_item = p.pool[end]
    if dist >= last_item.dist
        # p.k == length(p.pool) but item.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but item.dist improves the result set
    @inbounds p.pool[end] = Item(objID, dist)
    fix_order!(p)
    return true
end

"""
return the first item of the result set, the closest item
"""
function first(p::KnnResult{T}) where T
    @inbounds return p.pool[1]
end

"""
returns the last item of the result set
"""
function last(p::KnnResult{T}) where T
    @inbounds return p.pool[end]
end

"""
apply shift!(p.pool), an O(length(p.pool)) operation
"""
function shift!(p::KnnResult{T}) where T
    shift!(p.pool)
end

"""
apply pop!(p), an O(1) operation
"""
function pop!(p::KnnResult{T}) where T
    return pop!(p.pool)
end

"""
length returns the number of items in the result set
"""
Base.length(p::KnnResult{T}) where T = length(p.pool)

"""
The maximum allowed cardinality (the k of knn)
"""
maxlength(p::KnnResult{T}) where T = p.k

"""
covrad returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
function covrad(p::KnnResult{T})::Float64 where T
    return length(p.pool) < p.k ? typemax(Float64) : last(p).dist
end

function clear!(p::KnnResult{T}) where T
    resize!(p.pool, 0)
end

##### iterator interface
### KnnResult
function start(p::KnnResult{T}) where T
    return 1
end

function done(p::KnnResult{T}, state) where T
    return state > length(p)
end

function next(p::KnnResult{T}, state) where T
    return (p.pool[state], state + 1)
end

