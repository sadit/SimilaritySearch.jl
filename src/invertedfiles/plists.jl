# This file is part of InvertedFiles.jl

"""
    struct PostingList

A single posting list: the (sorted) identifiers of every object containing token
`tokenID`, plain ids with no associated weight (`InvertedFile` never needs one -- see
[`identiterator`](@ref)).
"""
struct PostingList{ListType<:AbstractVector}
    list::ListType
    tokenID::UInt32
end

@inline Base.length(plist::PostingList) = length(plist.list)

@inline Base.@propagate_inbounds Base.getindex(plist::PostingList{Vector{UInt32}}, i::Integer)::UInt32 = plist.list[i]
@inline Base.@propagate_inbounds Base.getindex(plist::PostingList{<:SubArray{UInt32}}, i::Integer)::UInt32 = plist.list[i]
