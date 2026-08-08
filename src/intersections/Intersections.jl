# This file is part of Intersections.jl

module Intersections
@inline Base.@propagate_inbounds function swapitems!(arr, i::Integer, j::Integer)
    @inbounds tmp = arr[i]
    @inbounds arr[i] = arr[j]
    @inbounds arr[j] = tmp
end

@inline Base.@propagate_inbounds function onmatch2!(output::Vector, A, i::Integer, B, j::Integer)
    push!(output, A[i])
end

@inline Base.@propagate_inbounds function onmatch!(output::Vector, L, P, m::Integer)
    push!(output, L[1][P[1]])
end

include("search.jl")
include("by.jl")
include("umerge.jl")
include("imerge.jl")
include("svs.jl")
include("bk.jl")
include("bkt.jl")
include("xmerge.jl")
end
