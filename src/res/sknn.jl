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


export SlugKnnResult

mutable struct SlugKnnResult <: Result
    pool::Vector{Item}
    sp::Int32 # start pos
    ep::Int32  # end pos
end

function fromjson(::Type{SlugKnnResult}, dict)
    SlugKnnResult([fromjson(Item, d) for d in dict["pool"]], dict["sp"], dict["ep"])
end

function SlugKnnResult(k::Int)
    c = SlugKnnResult(Vector{Item}(2k), 1, 1)
    clear!(c)
    c
end

function clear!(p::SlugKnnResult)
    p.sp = 1
    p.ep = 0
end

function normalize_range!(p::SlugKnnResult)
    if p.ep > 0 && p.ep == length(p.pool)
        n = length(p)
        l = p.sp
        for i=1:n
            @inbounds p.pool[i] = p.pool[l]
            l += 1
        end
        p.sp = 1
        p.ep = n
    end
end

"""
fix_order! fixes the sorted state of the array. It implements a kind of insertion sort.
It is efficient because of the insertion distribution
(with high probability, the new items are further the olders)
"""

function fix_order!(res::SlugKnnResult)
    arr::Vector{Item} = res.pool
    item = arr[res.ep]
    i = res.ep
    @inbounds while i > res.sp
        if item.dist < arr[i-1].dist
            arr[i] = arr[i-1]
        else
            arr[i] = item
            return
        end
        i -= 1
    end
    @inbounds arr[res.sp] = item
end


"""
push! appends an item to the end of the result set
"""

push!(p::SlugKnnResult, objID::I, dist::F) where {I <: Integer, F <: Real} = push!(p, convert(Int32, objID), convert(Float32, dist))

function push!(p::SlugKnnResult, objID::Int32, dist::Float32)

    n = length(p)

    if n == 0   # handling as special case to improve speed
        clear!(p)  # normalize the empty set
        p.ep += 1
        @inbounds p.pool[p.ep] = Item(objID, dist)
        return true
    end

    @inbounds if n < maxlength(p)  # fewer items than the maximum capacity
        normalize_range!(p)  # normalizes the populated range if needed
        p.ep += 1
        @inbounds p.pool[p.ep] = Item(objID, dist)
        fix_order!(p)
        return true
    end

    @inbounds last_item = p.pool[p.ep]
    if dist >= last_item.dist
        # maxlength(p) == length(p) but the new item.dist doesn't reduce the pool's radius
        return false
    end

    # maxlength(p) == length(p) -> item.dist improves the result set
    @inbounds p.pool[p.ep] = Item(objID, dist)
    fix_order!(p)
    return true
end

"""
return the first item of the result set, the closest item
"""
function first(p::SlugKnnResult)
    return p.pool[p.sp]
end

"""
returns the last item of the result set
"""
function last(p::SlugKnnResult)
    return p.pool[p.ep]
end

"""
apply shift!(p.pool), an O(1) operation
"""
function shift!(p::SlugKnnResult)
    @inbounds item = p.pool[p.sp]
    p.sp += 1
    return item
end

"""
apply pop!(p), an O(1) operation
"""
function pop!(p::SlugKnnResult)
    @inbounds item = p.pool[p.ep]
    p.ep -= 1
    return item
end

"""
length returns the number of items in the result set
"""
function length(p::SlugKnnResult)
    p.ep - p.sp + 1
end

function maxlength(p::SlugKnnResult)
    length(p.pool) >> 1
end

"""
covrad returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
function covrad(p::SlugKnnResult)::Float32
    return length(p) < maxlength(p) ? typemax(Float32) : last(p).dist
end


##### iterator interface
### SlugKnnResult
function start(p::SlugKnnResult)
    p.sp
end

function done(p::SlugKnnResult, state)
    state > p.ep
end

function next(p::SlugKnnResult, state)
    (p.pool[state], state + 1)
end
