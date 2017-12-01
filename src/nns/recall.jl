#  Copyright 2016,2017 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

export recall, recallByDist

function equals(a::Integer, b::Integer)
    return a == b
end

function equals(a::AbstractFloat, b::AbstractFloat)
    return abs(a - b) <= 1e-6  # eps(typeof(a))
end

function generic_recall(a::AbstractVector{T}, b::AbstractVector{T}; shift::Int=0) where {T <: Real}
    ai::Int = 1
    bi::Int = 1
    matches::Int = 0
    while ai <= length(a) && bi <= length(b)
        aitem = a[ai]
        bitem = b[bi]
        if equals(aitem, bitem)
            ai += 1
            bi += 1
            matches += 1
        elseif aitem < bitem
            ai += 1
        else
            bi += 1
        end
    end
    return (matches-shift) / (length(a)-shift)
end

function recall(a::AbstractVector{T}, b::AbstractVector{T}; shift::Int=0) where {T <: Real}
    A = [item for item in a]
    B = [item for item in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B, shift=shift)
end

function recall(a::AbstractVector{Item}, b::AbstractVector{Item}; shift::Int=0)
    recall([item.objID for item in a], [item.objID for item in b], shift=shift)
end

function recall(a::Result, b::Result; shift::Int=0)
    recall(collect(Item, a), collect(Item, b), shift=shift)
end

function recallByDist(a::AbstractVector{T}, b::AbstractVector{T}; shift::Int=0) where {T <: Real}
    A = [d for d in a]
    B = [d for d in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B, shift=shift)
end

function recallByDist(a::AbstractVector{Item}, b::AbstractVector{Item}; shift::Int=0)
    recallByDist([item.dist for item in a], [item.dist for item in b], shift=shift)
end

function recallByDist(a::Result, b::Result; shift::Int=0)
    recallByDist(collect(Item, a), collect(Item, b), shift=shift)
end
