# This file is part of InvertedFiles.jl

export SortedIntSet

struct SortedIntSet{VecInt}
    set::VecInt
end

@inline Base.eltype(::SortedIntSet{VecInt}) where VecInt = eltype(VecInt)
@inline Base.size(s::SortedIntSet) = size(s.set)
@inline Base.length(s::SortedIntSet) = length(s.set)
@inline Base.eachindex(s::SortedIntSet) = eachindex(s.set)
@inline Base.getindex(s::SortedIntSet, i::Integer) = @inbounds s.set[i]
@inline Base.first(s::SortedIntSet) = @inbounds s[1]
@inline Base.last(s::SortedIntSet) = @inbounds s[end]
