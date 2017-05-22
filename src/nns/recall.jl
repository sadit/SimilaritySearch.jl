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

export recall, recallByDist

function equals(a::Integer, b::Integer)
    return a == b
end

function equals(a::AbstractFloat, b::AbstractFloat)
    return abs(a - b) <= eps(typeof(a))
end

function generic_recall{T <: Real}(a::Array{T,1}, b::Array{T,1}, shiftK::Int=0)
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
    return (matches-shiftK) / (length(a)-shiftK)
end

function recall{T <: Real}(a::Array{T,1}, b::Array{T,1})
    A = [item for item in a]
    B = [item for item in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B)
end

function recall(a::Array{Item,1}, b::Array{Item,1})
    A = [item.objID for item in a]
    B = [item.objID for item in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B)
end


function recall(a::Result, b::Result)
    A = [item.objID for item in a]
    B = [item.objID for item in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B)
end

function recallByDist{T <: Real}(a::Array{T,1}, b::Array{T,1})
    A = [d for d in a]
    B = [d for d in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B)
end

function recallByDist(a::Array{Item,1}, b::Array{Item,1})
    A = [item.dist for item in a]
    B = [item.dist for item in b]
    sort!(A)
    sort!(B)
    generic_recall(A, B)
end

function recallByDist(a::Result, b::Result)
    A = [item.dist for item in a]
    B = [item.dist for item in b]
    # sort!(A)
    # sort!(B)
    generic_recall(A, B)
end
