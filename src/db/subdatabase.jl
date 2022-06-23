# This file is a part of SimilaritySearch.jl

# SubDatabase ~ view of the dataset
#
struct SubDatabase{DBType,RType} <: AbstractDatabase
    db::DBType
    map::RType
end

@inline Base.getindex(sdb::SubDatabase, i::Integer) = @inbounds sdb.db[sdb.map[i]]
@inline Base.length(sdb::SubDatabase) = length(sdb.map)
@inline Base.eachindex(sdb::SubDatabase) = eachindex(sdb.map)
@inline function Base.push!(sdb::SubDatabase, v)
    error("push! unsupported operation on SubDatabase")
end
@inline Base.eltype(sdb::SubDatabase) = eltype(sdb.db)
@inline Random.rand(db::SubDatabase, n::Integer) = SubDatabase(db.db, rand(db.map, n))