# This file is a part of SimilaritySearch.jl


@inline function _bitindices(i_::Integer)
    i = convert(UInt64, i_) - one(UInt64)
    (i >>> 6) + 1, (i & 63)
end

@inline function visited(vstate::Vector{UInt64}, i_::Integer)
    b, i = _bitindices(i_)
    @inbounds ((vstate[b] >>> i) & one(UInt64)) == one(UInt64)
end

@inline function visit!(vstate::Vector{UInt64}, i_::Integer)
    b, i = _bitindices(i_)
    @inbounds vstate[b] |= (one(UInt64) << i)
end

@inline function check_visited_and_visit!(vstate::Vector{UInt64}, i_::Integer)
    b, i = _bitindices(i_)
    v = ((vstate[b] >>> i) & one(UInt64)) == one(UInt64)
    !v && ( @inbounds vstate[b] |= (one(UInt64) << i) )
    v
end

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

# const GlobalVisitedVertices = [BitArray(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Vector{UInt8}(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Set{Int32}()]  # initialized at __init__ function
const GlobalVisitedVertices = [Vector{UInt64}(undef, 32)]

function __init__visitedvertices()
    for i in 2:Threads.nthreads()
        push!(GlobalVisitedVertices, copy(GlobalVisitedVertices[1]))
    end

    #=for i in eachindex(GlobalVisitedVertices)
        sizehint!(GlobalVisitedVertices[i], 128)
    end=#
end

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

@inline function _init_vv(v::Vector{UInt64}, n)
    n64 = 1 + ((n-1) >>> 6)
    n64 > length(v) && resize!(v, n64)
    @inbounds for i in 1:n64
        v[i] = 0
    end
    v
end