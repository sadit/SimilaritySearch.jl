# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base:
    push!, popfirst!, pop!, length, last, first, empty!

export Item, KnnResult, maxlength, covrad, reset!

struct Item
    objID::Int32
    dist::Float32
end

mutable struct KnnResult #{T}
    k::Int32
    pool::Vector{Item}
    #    pool::Vector{Item{T}}

    function KnnResult(k::Integer)
        v = Vector{Item}()
        sizehint!(v, k)
        new(k, v)
    end
 
end

"""
    fix_order!(res::KnnResult)

Fixes the sorted state of the array. It implements a kind of insertion sort
It is efficient due to the expected distribution of the items being inserted
(few smaller than the ones already inside)
"""
@inline function fix_order!(res::KnnResult) 
    arr = res.pool
    item = arr[end]
    i = length(arr)
    @inbounds while i > 1
        if item.dist < arr[i-1].dist
            arr[i] = arr[i-1]
        else
            arr[i] = item
            return nothing
        end
        i -= 1
    end

    arr[1] = item
    nothing
end

"""
    push!(p::KnnResult, objID::Integer, dist::Number)

Appends an item into the result set
"""
#push!(p::KnnResult{Int64}, objID::I, dist::F) where {I <: Union{Int32,Int16}, F <: Real} = push!(p, convert(Int64, objID), convert(Float64, dist))
#push!(p::KnnResult{T}, objID::T, dist::F) where {T, F <: Union{Float16, Float32}} = push!(p, objID, convert(Float64, dist))

function push!(p::KnnResult, objID::Integer, dist::Number)
    if length(p.pool) < p.k
        # fewer items than the maximum capacity
        push!(p.pool, Item(convert(Int32, objID), convert(Float32, dist)))
        #push!(p.pool, Item(objID, dist))
        fix_order!(p)
        return true
    end

    if dist >= last(p).dist
        # p.k == length(p.pool) but item.dist doesn't improve the pool's radius
        return false
    end

    # p.k == length(p.pool) but item.dist improves the result set
    @inbounds p.pool[end] = Item(objID, dist)
    fix_order!(p)
    true
end


"""
    first(p::KnnResult)

Return the first item of the result set, the closest item
"""
function first(p::KnnResult)
    p.pool[1]
end

"""
    last(p::KnnResult) 

Returns the last item of the result set
"""
function last(p::KnnResult) 
    p.pool[end]
end

"""
    popfirst!(p::KnnResult)

Removes and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation
"""
function popfirst!(p::KnnResult)
    popfirst!(p.pool)
end

"""
    pop!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
function pop!(p::KnnResult)
    pop!(p.pool)
end

"""
    length(p::KnnResult)

length returns the number of items in the result set
"""
Base.length(p::KnnResult) = length(p.pool)

"""
    maxlength(p::KnnResult)

The maximum allowed cardinality (the k of knn)
"""
maxlength(p::KnnResult) = p.k

"""
    covrad(p::KnnResult)::Float64

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
function covrad(p::KnnResult)::Float64
    return length(p.pool) < p.k ? typemax(Float32) : last(p).dist
end

"""
    empty!(p::KnnResult)

Clears the content of the result pool
"""
function empty!(p::KnnResult)
    empty!(p.pool)
end

function reset!(p::KnnResult, k::Integer)
    empty!(p)
    p.k = k
    p
end

##### iterator interface
### KnnResult
function Base.iterate(p::KnnResult)
    return length(p) == 0 ? nothing : (first(p), 2)
end

function Base.iterate(p::KnnResult, state::Int)
    if state > length(p)
        return nothing
    end

    return p.pool[state], state + 1
end
