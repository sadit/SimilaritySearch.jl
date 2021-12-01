# This file is a part of SimilaritySearch.jl

# Database interface
using Random, StrideArrays
export AbstractDatabase, MatrixDatabase, VectorDatabase, SubDatabase, StrideDatabase

abstract type AbstractDatabase
end

struct StrideDatabase{DataType} <: AbstractDatabase
    X::DataType
end
StrideDatabase(m::Matrix) = StrideDatabase(StrideArray(m))

@inline function Base.getindex(db::StrideDatabase, i::Integer)
    @view db.X[:, i]
end

@inline function Base.getindex(db::StrideDatabase, lst::Vector{<:Integer})
    [db[i] for i in lst]
end

@inline Base.length(db::StrideDatabase) = size(db.X, 2)
@inline Base.eachindex(db::StrideDatabase) = 1:length(db)
@inline Base.eltype(db::StrideDatabase) = eltype(db.X)

#
struct MatrixDatabase{DType,Dim} <: AbstractDatabase
    data::Vector{DType}
    MatrixDatabase(m::Matrix) = new{eltype(m), size(m, 1)}(vec(m))
    MatrixDatabase(dtype::Type, dim::Integer) = new{dtype,dim}(Vector{dtype}(undef, 0))
    MatrixDatabase(A::AbstractVector) = new{eltype(A[1]), length(A[1])}(vcat(A...))
end

@inline function Base.getindex(db::MatrixDatabase{DType,dim}, i::Integer) where {DType,dim}
    ep = i * dim
    @view db.data[(ep - dim + 1):ep]
end

@inline function Base.getindex(db::MatrixDatabase{DType,dim}, lst::Vector{<:Integer}) where {DType,dim}
    [db[i] for i in lst]
end

@inline Base.length(db::MatrixDatabase{DType,dim}) where {DType,dim} = Int(length(db.data) / dim)
@inline Base.eachindex(db::MatrixDatabase) = 1:length(db)
@inline function Base.push!(db::MatrixDatabase{DType,dim}, v) where {DType,dim}
    append!(db.data, v)
    db
end
@inline Base.append!(a::MatrixDatabase{DType,dim}, b::MatrixDatabase{DType2,dim})  where {DType,dim,DType2} = append!(a.data, b.data)
@inline function Base.append!(a::MatrixDatabase{DType,dim}, B) where {DType,dim}
    @assert all(b->length(b) == dim, B)
    for b in B
        push!(a, b)
    end
end

@inline Base.eltype(db::MatrixDatabase{DType,dim}) where {DType,dim} = AbstractVector{DType}
@with_kw struct VectorDatabase{DType} <: AbstractDatabase
    data::Vector{DType} = Vector{Vector{Float32}}(undef, 0)
end

VectorDatabase(m::Matrix) = VectorDatabase([Vector(c) for c in eachcol(m)])

@inline Base.getindex(db::VectorDatabase, i) = @inbounds db.data[i]
@inline Base.length(db::VectorDatabase) = length(db.data)
@inline Base.eachindex(db::VectorDatabase) = eachindex(db.data)
@inline function Base.push!(db::VectorDatabase, v)
    push!(db.data, v)
    db
end
@inline Base.append!(a::VectorDatabase, b::VectorDatabase) = append!(a.data, b.data)
@inline Base.append!(a::VectorDatabase, b) = append!(a.data, b)
@inline Base.eltype(db::VectorDatabase) = eltype(db.data)


function Base.iterate(db::AbstractDatabase, state::Int=1)
    n = length(db)
    if n == 0 || state > n
        nothing
    else
        @inbounds db[state], state + 1
    end
end

struct SubDatabase{DBType,RType} <: AbstractDatabase
    db::DBType
    range_::RType
end

@inline Base.getindex(sdb::SubDatabase, i) = @inbounds sdb.db[sdb.range_[i]]
@inline Base.length(sdb::SubDatabase) = length(sdb.range_)
@inline Base.eachindex(sdb::SubDatabase) = eachindex(sdb.range_)
@inline function Base.push!(sdb::SubDatabase, v)
    error("push! unsupported operation on SubDatabase")
end
@inline Base.eltype(sdb::SubDatabase) = eltype(sdb.db)

@inline Base.view(db::AbstractDatabase, range_) = SubDatabase(db, range_)
@inline Base.size(db::AbstractDatabase) = (length(db),)
@inline Random.rand(db::AbstractDatabase) = @inbounds db[rand(eachindex(db))]
@inline Random.rand(db::AbstractDatabase, n::Integer) = [rand(db) for i in 1:n]