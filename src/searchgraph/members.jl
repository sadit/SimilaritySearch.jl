# This file is a part of SimilaritySearch.jl

export Members, members, representative, ismember

"""
    Members()

The near-duplicate members of a [`SearchGraph`](@ref). When a graph is built with
`Neighborhood(neardup=ϵ)`, an object whose nearest indexed object lies within `ϵ` does not
become a node of its own: it is a *member* of that object's cluster, its adjacency is the
single edge to the cluster's representative, no node links to it, and the search never
visits it. The search therefore answers with representatives -- one per cluster -- and
[`expand`](@ref)/[`expand!`](@ref) turn those into the raw neighbors through this table.
Members still count in `length(G)` and keep their database ids; only the navigation changes.

Two maps, kept consistent by [`addmember!`](@ref): `lists` gives a representative's members
and `representative` a member's representative. A representative is never a member.
"""
struct Members
    lists::Dict{UInt32,Vector{UInt32}}
    representative::Dict{UInt32,UInt32}
end

Members() = Members(Dict{UInt32,Vector{UInt32}}(), Dict{UInt32,UInt32}())

"The number of members, i.e. of indexed objects that are not nodes."
Base.length(M::Members) = length(M.representative)
Base.isempty(M::Members) = isempty(M.representative)

const _NOMEMBERS = UInt32[]

"""
    ismember(G, id) -> Bool

Whether `id` is a near-duplicate member of a cluster (see [`Members`](@ref)). Every index
answers it; only a [`SearchGraph`](@ref) built with `Neighborhood(neardup=ϵ)` has members.
"""
ismember(M::Members, id::Integer) = haskey(M.representative, UInt32(id))

"""
    representative(G, id) -> UInt32

The representative of `id`'s near-duplicate cluster: `id` itself when it is a node (see
[`Members`](@ref)).
"""
representative(M::Members, id::Integer) = get(M.representative, UInt32(id), UInt32(id))

"""
    members(G, id) -> AbstractVector{UInt32}

The members of the cluster `id` represents: empty for a node without members, and for a
member itself -- ask its [`representative`](@ref) (see [`Members`](@ref)).
"""
members(M::Members, id::Integer) = get(M.lists, UInt32(id), _NOMEMBERS)

ismember(::AbstractSearchIndex, id::Integer) = false
representative(::AbstractSearchIndex, id::Integer) = UInt32(id)
members(::AbstractSearchIndex, id::Integer) = _NOMEMBERS

"""
    addmember!(M::Members, rep, id)

Registers `id` as a member of `rep`, which must be a node (not a member itself): the callers
resolve chains before calling. A member is never registered twice.
"""
function addmember!(M::Members, rep::Integer, id::Integer)
    rep = UInt32(rep); id = UInt32(id)
    rep == id && throw(ArgumentError("addmember!: $id cannot be its own representative"))
    ismember(M, rep) && throw(ArgumentError("addmember!: $rep is a member of $(representative(M, rep)) and cannot represent $id; resolve first"))
    haskey(M.representative, id) && throw(ArgumentError("addmember!: $id is already a member of $(M.representative[id])"))
    M.representative[id] = rep
    push!(get!(M.lists, rep, UInt32[]), id)
    M
end

"The ids of `id`'s whole cluster: its representative first, then the members."
function clusterids(G, id::Integer)
    r = representative(G, id)
    (r, members(G, r)...)
end

function Base.show(io::IO, M::Members)
    print(io, "Members(", length(M), " members in ", length(M.lists), " clusters)")
end
