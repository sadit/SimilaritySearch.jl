# This file is a part of SimilaritySearch.jl

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

@inline getstate(vstate::Vector{UInt8}, i) = @inbounds vstate[i]

@inline function setstate!(vstate::Vector{UInt8}, i, state)
    @inbounds vstate[i] = state
end

@inline getstate(vstate::Dict{Int32,UInt8}, i) = @inbounds get(vstate, i, UNKNOWN)

@inline function setstate!(vstate::Dict{Int32,UInt8}, i, state)
    @inbounds vstate[i] = state
end

@inline function visit!(vstate, visited)
    for v in visited
        setstate!(vstate, v, VISITED)
    end
end

const GlobalVisitedVertices = [Vector{UInt8}(undef, 1)]  # initialized at __init__ function
# const GlobalVisitedVertices = [Dict{Int32,UInt8}()]  # initialized at __init__ function

function __init__visitedvertices()
    for i in 2:Threads.nthreads()
        push!(GlobalVisitedVertices, copy(GlobalVisitedVertices[1]))
    end

    #=for i in eachindex(GlobalVisitedVertices)
        sizehint!(GlobalVisitedVertices[i], 128)
    end=#
end

@inline function _init_vv(v::Vector, n)
    length(v) < n && resize!(v, n)
    fill!(v, 0)
    v
end

@inline function _init_vv(v::Dict, n)
    empty!(v)
    v
end

@inline function getvisitedvertices(index::SearchGraph)
    @inbounds v = GlobalVisitedVertices[Threads.threadid()]
    _init_vv(v, length(index))
end