# This file is a part of SimilaritySearch.jl

@inline function _b64indices(i_::UInt64)
    i = i_ - one(UInt64)
    (i >>> 6) + 1, (i & 63)
end

@inline _b64block(i_::UInt64) = ((i_ - one(UInt64)) >>> 6) + 1

#=struct Vector{UInt64}Bits
    B::Vector{UInt64}
end=#


function reuse!(v::Vector{UInt64}, n::Integer)
    let n = convert(UInt64, n), m = _b64block(n), B = v
        m > length(B) && resize!(B, m)
        fill!(B, zero(UInt64))
        v
    end
end

@inline function visited(vstate, i_::UInt64)::Bool
    b, i = _b64indices(i_)
    @inbounds (vstate[b] >>> i) & one(UInt64)
end

@inline function visit!(vstate, i_::UInt64)
    b, i = _b64indices(i_)
    @inbounds vstate[b] |= (one(UInt64) << i)
    nothing
end

@inline function check_visited_and_visit!(vstate, i_::Integer)::Bool
    b, i = _b64indices(i_)
    v = Bool((vstate[b] >>> i) & one(UInt64))
    !v && (@inbounds vstate[b] |= (one(UInt64) << i))
    v
end

#### Vector{UInt64} with int sets
#=
function Vector{UInt64}Set(n::Integer)
    v = Set{UInt32}()
    sizehint!(v, n)
    v
end

@inline visited(vstate::Set{UInt32}, i::Integer)::Bool = i ∈ vstate
@inline visit!(vstate::Set{UInt32}, i::Integer) = push!(vstate, i)
@inline function check_visited_and_visit!(vstate::Set{UInt32}, i::Integer)
    v = visited(vstate, i)
    !v && visit!(vstate, i)
    v
end

function reuse!(v::Set{UInt32}, n::Integer)
    empty!(v)
    sizehint!(v, n)
    v
end
=#
