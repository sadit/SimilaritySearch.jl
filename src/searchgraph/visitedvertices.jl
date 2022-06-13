# This file is a part of SimilaritySearch.jl

@inline visited(vstate::BitVector, i) = @inbounds vstate[i]

@inline function visit!(vstate::BitVector, i)
    @inbounds vstate[i] = true
end

@inline visited(vstate::Vector{UInt8}, i)::Bool = @inbounds vstate[i] == 1

@inline function visit!(vstate::Vector{UInt8}, i, state)
    @inbounds vstate[i] = 1
end

@inline visited(vstate::Set{Int32}, i)::Bool = i ∈ vstate

@inline function visit!(vstate::Set{Int32}, i)
    @inbounds push!(vstate, i)
end

const GlobalVisitedVertices = [BitArray(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Vector{UInt8}(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Set{Int32}()]  # initialized at __init__ function

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
