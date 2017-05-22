export NnResult

mutable struct NnResult <: Result
    item::Item
end

NnResult() = return NnResult(Item(-1, -1))

function fromjson(::Type{NnResult}, dict)
    NnResult(fromjson(Item, dict["item"]))
end

push!(p::NnResult, objID::I, dist::F) where {I <: Integer, F <: Real} = push!(p, convert(Int32, objID), convert(Float32, dist))
function push!(p::NnResult, objID::Int32, dist::Float32)
    if p.item.objID == -1
        p.item = Item(objID, dist)
    elseif dist >= p.item.dist
        return false
    else
        p.item = Item(objID, dist)
    end

    return true
end

"""
return the first item of the result set, the closest item
"""

function first(p::NnResult)
    return p.item
end

"""
returns the last item of the result set
"""
function last(p::NnResult)
    return p.item
end

"""
apply shift!(p.pool), an O(length(p.pool)) operation
"""
function shift!(p::NnResult)
    item = p.item
    clear!(p)
    item
end

"""
apply pop!(p), an O(1) operation
"""
function pop!(p::NnResult)
    item = p.item
    clear!(p)
    item
end

"""
length returns the number of items in the result set
"""

function length(p::NnResult)::Int
    return p.item.objID == -1 ? 0 : 1
end

function maxlength(p::NnResult)
    return 1
end

"""
covrad returns the coverage radius of the result set; if length(p) < K then typemax(Float64) is returned
"""
function covrad(p::NnResult)::Float32
    return length(p) == 0 ? typemax(Float32) : p.item.dist
end

function clear!(p::NnResult)
    p.item = Item(-1, 0f0)
end

##### iterator interface
### NnResult
function start(p::NnResult)
    return length(p)
end

function done(p::NnResult, state)
    if state == 1
        return false
    else
        return true
    end
end

function next(p::NnResult, state)
    return (p.item, 0)
end


### both
function eltype(p::Result)
    return Item
end
