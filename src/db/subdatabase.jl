# This file is a part of SimilaritySearch.jl

# SubDatabase ~ view of the dataset
#
struct SubDatabase{DBType<:AbstractDatabase,RType} <: AbstractDatabase
    parent::DBType
    map::RType
end

function show(io::IO, db::SubDatabase; prefix="", indent="  ")
    println(io, prefix, "SubDatabase:")
    prefix = prefix * indent
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "length: ", length(db))
    println(io, prefix, "parent-type: ", typeof(db.parent))
end

@inline Base.getindex(S::SubDatabase, i::Integer) = @inbounds S.parent[S.map[i]]
@inline Base.length(S::SubDatabase) = length(S.map)
@inline Base.eachindex(S::SubDatabase) = eachindex(S.map)
@inline push_item!(S::SubDatabase, v) = error("push! unsupported operation on SubDatabase")
@inline Base.eltype(S::SubDatabase) = eltype(S.parent)
@inline Random.rand(S::SubDatabase, n::Integer) = SubDatabase(S.parent, rand(S.map, n))
