# This file is a part of SimilaritySearch.jl


#=struct Vector{UInt64}Bits
    B::Vector{UInt64}
end=#

@inline function _b64indices(i_::UInt64)
    i = i_ - one(UInt64)
    (i >>> 6) + 1, (i & 63)
end

@inline _b64block(i_::UInt64) = ((i_ - one(UInt64)) >>> 6) + 1

function reuse!(B::AbstractVector{UInt64}, n::Integer)
    n > 0 && let n = convert(UInt64, n), m = _b64block(n)
        m > length(B) && resize!(B, m)
        @inbounds @simd for i in 1:m
            B[i] = xor(B[i], B[i])
        end
    end

    B
end

"""
    check_visited_and_visit!(vstate, i_::Integer)::Bool

Checks that `i_` is already visited:
    - returns true if already visited
    - returns false if is not visited yet, but marks it as visited before return
"""
@inline function check_visited_and_visit!(vstate::AbstractVector{UInt64}, i_::Integer)::Bool
    b, i = _b64indices(i_)
    @inbounds v = Bool((vstate[b] >>> i) & one(UInt64))
    !v && (@inbounds vstate[b] |= (one(UInt64) << i))
    v
end

@inline function visited(vstate::AbstractVector{UInt64}, i_::UInt64)::Bool
    b, i = _b64indices(i_)
    @inbounds (vstate[b] >>> i) & one(UInt64)
end

@inline function visit!(vstate::AbstractVector{UInt64}, i_::UInt64)
    b, i = _b64indices(i_)
    @inbounds vstate[b] |= (one(UInt64) << i)
    nothing
end

#### Int set

function reuse!(v::Set{UInt32}, n::Integer)
    empty!(v)
    sizehint!(v, ceil(Int, sqrt(n)))
    v
end

@inline visited(vstate::Set{UInt32}, i::Integer)::Bool = i ∈ vstate
@inline visit!(vstate::Set{UInt32}, i::Integer) = push!(vstate, i)
@inline function check_visited_and_visit!(vstate::Set{UInt32}, i::Integer)
    v = visited(vstate, i)
    !v && visit!(vstate, i)
    v
end
