# This file is a part of SimilaritySearch.jl

const UNKNOWN = UInt8(0)
const VISITED = UInt8(1)
const EXPLORED = UInt8(2)

const VisitedVertices = Dict{Int32, UInt8}

function VisitedVertices(n)
    v = VisitedVertices()
    sizehint!(v, h)
    v
end

@inline getstate(vstate::VisitedVertices, i) = get(vstate, i, UNKNOWN)
@inline function setstate!(vstate::VisitedVertices, i, state)
    vstate[i] = state
end

@inline function visit!(vstate::VisitedVertices, visited)
    for v in visited
        setstate!(vstate, v, VISITED)
    end
end

const GlobalVisitedVertices = [VisitedVertices()]  # initialized at __init__ function

@inline getvisitedvertices(vstate=nothing) = vstate !== nothing ? vstate : @inbounds GlobalVisitedVertices[Threads.threadid()]
