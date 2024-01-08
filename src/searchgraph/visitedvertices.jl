# This file is a part of SimilaritySearch.jl

@inline function _bitindices(i_::UInt64)
    # i = convert(UInt64, i_) - one(UInt64)
    i = i_ - one(UInt64)
    (i >>> 6) + 1, (i & 63)
end

struct VisitedVerticesBits
    B::Vector{UInt64}
end

VisitedVerticesBits(n::Integer) = VisitedVerticesBits(Vector{UInt64}(undef, n))

Base.copy(v::VisitedVerticesBits) = VisitedVerticesBits(copy(v.B))

function reuse!(v::VisitedVerticesBits, n::Integer)
    let n = convert(UInt64, n)
        n64 = 1 + ((n-1) >>> 6)
        B = v.B
        n64 > length(B) && resize!(B, n64)
        z = zero(UInt64)
        @inbounds for i in 1:n64
            B[i] !== z && (B[i] = z)
        end
        v
    end
end

@inline function visited(vstate::VisitedVerticesBits, i_::UInt64)
    b, i = _bitindices(i_)
    @inbounds ((vstate.B[b] >>> i) & one(UInt64)) == one(UInt64)
end

@inline function visit!(vstate::VisitedVerticesBits, i_::UInt64)
    b, i = _bitindices(i_)
    @inbounds vstate.B[b] |= (one(UInt64) << i)
end

@inline function check_visited_and_visit!(vstate::VisitedVerticesBits, i_::Integer)
    b, i = _bitindices(i_)
    v = ((vstate.B[b] >>> i) & one(UInt64)) == one(UInt64)
    !v && ( @inbounds vstate.B[b] |= (one(UInt64) << i) )
    v
end

#=
@inline function check_visited_and_visit!(vstate, i::Integer)
    v = visited(vstate, i)
    !v && visit!(vstate, i)
    v
end

#### VisitedVertices with BitVector


@inline visited(vstate::BitVector, i::Integer)::Bool = @inbounds vstate[i]

@inline function visit!(vstate::BitVector, i)
    @inbounds vstate[i] = true
end

#### VisitedVertices with byte arrays

@inline visited(vstate::Vector{UInt8}, i::Integer)::Bool = @inbounds vstate[i] == 1

@inline function visit!(vstate::Vector{UInt8}, i::Integer)
    @inbounds vstate[i] = 1
end

#### VisitedVertices with int sets

@inline visited(vstate::Set{Int32}, i::Integer)::Bool = i ∈ vstate

@inline function visit!(vstate::Set{Int32}, i::Integer)
    @inbounds push!(vstate, i)
end
=#
# const GlobalVisitedVertices = [BitArray(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Vector{UInt8}(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Set{Int32}()]  # initialized at __init__ function

#=
@inline function _init_vv(v::AbstractVector, n)
    # length(v) < n &&
    resize!(v, n)
    fill!(v, 0)
    v
end

@inline function _init_vv(v::BitVector, n)
    # length(v) < n &&
    resize!(v, n)
    fill!(v.chunks, 0)
    v
end

@inline function _init_vv(v::Set, n)
    empty!(v)
    v
end
=#
