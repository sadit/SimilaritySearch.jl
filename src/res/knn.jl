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

import Base.push!, Base.shift!, Base.pop!, Base.length, Base.start, Base.done, Base.next, Base.eltype, Base.last, Base.first

export Item, KnnResult, push!, first, last, shift!, pop!, length, maxlength, covrad, clear!, start, done, next, start, done, next, eltype

struct Item
    objID::Int32
    dist::Float32
end

function save(ostream, item::Item)
    write(ostream, string(item.objID, ' ', item.dist, '\n'))
end

function load(istream, ::Type{Item})
    a, b = split(readline(istream), ' ')
    return Item(parse(Int32, a), parse(Float32, b))
end

include("nn.jl")
include("sknn.jl")

mutable struct KnnResult <: Result
    k::Int
    pool::Vector{Item}
end

function KnnResult(k::Int)
    v = Vector{Item}()
    sizehint!(v, k)
    KnnResult(k, v)
end

function load(istream, ::Type{KnnResult})::KnnResult
    a, b = split(readline(istream), ' ')
    k = parse(Int32, a)
    pop = parse(Int32, b)
    KnnResult(k, [load(istream, Item) for i in 1:pop])
end

function save(ostream, obj::KnnResult)
    write(ostream, string(obj.k, ' ', length(obj.pool), '\n'))
    for item in obj.pool
        save(ostream, item)
    end
end

"""
fix_order! fixes the sorted state of the array. It implements a kind of insertion sort
It is efficient due to the expected distribution of the items being inserted
(few smaller than the ones already inside)
"""
@inline function fix_order!(res::KnnResult)
    arr::Vector{Item} = res.pool

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
push!(p::KnnResult, objID::I, dist::F) where {I <: Integer, F <: Real} = push!(p, convert(Int32, objID), convert(Float32, dist))

function push!(p::KnnResult, objID::Int32, dist::Float32)
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
function first(p::KnnResult)
    @inbounds return p.pool[1]
end

"""
returns the last item of the result set
"""
function last(p::KnnResult)
    @inbounds return p.pool[end]
end

"""
apply shift!(p.pool), an O(length(p.pool)) operation
"""
function shift!(p::KnnResult)
    shift!(p.pool)
end

"""
apply pop!(p), an O(1) operation
"""
function pop!(p::KnnResult)
    return pop!(p.pool)
end

"""
length returns the number of items in the result set
"""
Base.length(p::KnnResult) = length(p.pool)

"""
The maximum allowed cardinality (the k of knn)
"""
maxlength(p::KnnResult) = p.k

"""
covrad returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
function covrad(p::KnnResult)::Float32
    return length(p.pool) < p.k ? typemax(Float32) : last(p).dist
end

function clear!(p::KnnResult)
    resize!(p.pool, 0)
end

##### iterator interface
### KnnResult
function start(p::KnnResult)
    return 1
end

function done(p::KnnResult, state)
    return state > length(p)
end

function next(p::KnnResult, state)
    return (p.pool[state], state + 1)
end
