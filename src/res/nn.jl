export NnResult

mutable struct NnResult <: Result
    item::Item
end

NnResult() = return NnResult(Item(-1, -1))

push!(p::NnResult, objID::I, dist::F) where {I <: Integer, F <: Real} = push!(p, convert(Int64, objID), convert(Float64, dist))
function push!(p::NnResult, objID::Int64, dist::Float64)
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

first(p::NnResult) = p.item

"""
returns the last item of the result set
"""
last(p::NnResult) = p.item

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

length(p::NnResult)::Int = p.item.objID == -1 ? 0 : 1
maxlength(p::NnResult) = 1

"""
covrad returns the coverage radius of the result set; if length(p) < K then typemax(Float64) is returned
"""
covrad(p::NnResult)::Float64 = (length(p) == 0 ? typemax(Float64) : p.item.dist)

function clear!(p::NnResult)
    p.item = Item(-1, 0f0)
end

##### iterator interface
### NnResult
start(p::NnResult) = length(p)
done(p::NnResult, state) = state != 1
next(p::NnResult, state) = (p.item, 0)

### both
eltype(p::Result) = Item

