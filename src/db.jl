# This file is a part of SimilaritySearch.jl

# Database interface
using Random, StrideArrays
export AbstractDatabase, MatrixDatabase, DynamicMatrixDatabase, VectorDatabase, SubDatabase, StrideDatabase

abstract type AbstractDatabase
end

#
# Support for StrideArrays
#
struct StrideDatabase{DataType} <: AbstractDatabase
    X::DataType
end
StrideDatabase(m::Matrix) = StrideDatabase(StrideArray(m))

@inline function Base.getindex(db::StrideDatabase, i::Integer)
    @view db.X[:, i]
end
@inline Base.setindex!(db::StrideDatabase, value, i) = @inbounds (db.data[i] .= value)

@inline function Base.getindex(db::StrideDatabase, lst::Vector{<:Integer})
    [db[i] for i in lst]
end

@inline Base.length(db::StrideDatabase) = size(db.X, 2)
@inline Base.eachindex(db::StrideDatabase) = 1:length(db)
@inline Base.eltype(db::StrideDatabase) = eltype(db.X)

#
# Matrix dataset, i.e., columns are objects. We store them as vector to support appending items
#
struct DynamicMatrixDatabase{DType,Dim} <: AbstractDatabase
    data::Vector{DType}
    DynamicMatrixDatabase(m::Matrix) = new{eltype(m), size(m, 1)}(vec(m))
    DynamicMatrixDatabase(dtype::Type, dim::Integer) = new{dtype,dim}(Vector{dtype}(undef, 0))
    DynamicMatrixDatabase(A::AbstractVector) = new{eltype(A[1]), length(A[1])}(vcat(A...))
end

@inline function Base.getindex(db::DynamicMatrixDatabase{DType,dim}, i::Integer) where {DType,dim}
    ep = i * dim
    @inbounds @view db.data[(ep - dim + 1):ep]
end
@inline Base.setindex!(db::DynamicMatrixDatabase, value, i) = @inbounds (db.data[i] .= value)

@inline function Base.getindex(db::DynamicMatrixDatabase{DType,dim}, lst::Vector{<:Integer}) where {DType,dim}
    [db[i] for i in lst]
end

@inline Base.length(db::DynamicMatrixDatabase{DType,dim}) where {DType,dim} = length(db.data) ÷ dim
@inline Base.eachindex(db::DynamicMatrixDatabase) = 1:length(db)
@inline function Base.push!(db::DynamicMatrixDatabase{DType,dim}, v) where {DType,dim}
    append!(db.data, v)
    db
end
@inline Base.append!(a::DynamicMatrixDatabase{DType,dim}, b::DynamicMatrixDatabase{DType2,dim})  where {DType,dim,DType2} = append!(a.data, b.data)
@inline function Base.append!(a::DynamicMatrixDatabase{DType,dim}, B) where {DType,dim}
    @assert all(b->length(b) == dim, B)
    for b in B
        push!(a, b)
    end
end

@inline Base.eltype(db::DynamicMatrixDatabase{DType,dim}) where {DType,dim} = AbstractVector{DType}

#
# Matrix dataset, i.e., columns are objects. We store them as vector to support appending items
#
struct MatrixDatabase{MType} <: AbstractDatabase
    matrix::MType
    MatrixDatabase(m::Matrix) = new{typeof(m)}(m)
end

@inline Base.getindex(db::MatrixDatabase, i::Integer) = view(db.matrix, :, i)
@inline Base.setindex!(db::MatrixDatabase, value, i) = @inbounds (db.matrix[:, i] .= value)
@inline Base.length(db::MatrixDatabase) = size(db.matrix, 2)
@inline Base.eachindex(db::MatrixDatabase) = 1:length(db)
@inline Base.push!(db::MatrixDatabase, v) = error("Unsupported operation")
@inline Base.append!(a::MatrixDatabase, b) = error("Unsupported operation")
@inline Base.eltype(db::MatrixDatabase) = AbstractVector{eltype(db.matrix)}

#
# Generic array of objects
#
struct VectorDatabase{DataType} <: AbstractDatabase
    data::Vector{DataType}  # by default, we use array of float32 vectors
end

VectorDatabase(t::Type=Vector{Float32}) = VectorDatabase(Vector{t}(undef, 0))
VectorDatabase(M::Matrix) = VectorDatabase([Vector(c) for c in eachcol(M)])
VectorDatabase(M::VectorDatabase) = VectorDatabase(M.data)
#VectorDatabase(M::AbstractDatabase) = VectorDatabase([Vector(m) for m in M])

Base.convert(::Type{AbstractDatabase}, M::AbstractDatabase) = M
Base.convert(::Type{AbstractDatabase}, M::Matrix) = MatrixDatabase(M)
Base.convert(::Type{AbstractDatabase}, M::Vector) = VectorDatabase(M)
Base.convert(::Type{AbstractDatabase}, M::Vector{Any}) = VectorDatabase(typeof(first(M)).(M))
Base.convert(::Type{AbstractDatabase}, M::AbstractVector) = VectorDatabase(typeof(first(M)).(M))
Base.convert(::Type{<:AbstractVector}, M::VectorDatabase{T}) where T = M.data
Base.convert(::Type{<:AbstractVector}, M::AbstractDatabase) = collect(M)

@inline Base.getindex(db::VectorDatabase, i) = @inbounds db.data[i]
@inline Base.setindex!(db::VectorDatabase, value, i) = @inbounds (db.data[i] = value)
@inline Base.length(db::VectorDatabase) = length(db.data)
@inline Base.eachindex(db::VectorDatabase) = eachindex(db.data)
@inline function Base.push!(db::VectorDatabase, v)
    push!(db.data, v)
    db
end
@inline Base.append!(a::VectorDatabase, b::VectorDatabase) = append!(a.data, b.data)
@inline Base.append!(a::VectorDatabase, b) = append!(a.data, b)
@inline Base.eltype(db::VectorDatabase) = eltype(db.data)

#
# SubDatabase ~ view of the dataset
#
struct SubDatabase{DBType,RType} <: AbstractDatabase
    db::DBType
    map::RType
end

@inline Base.getindex(sdb::SubDatabase, i) = @inbounds sdb.db[sdb.map[i]]
@inline Base.length(sdb::SubDatabase) = length(sdb.map)
@inline Base.eachindex(sdb::SubDatabase) = eachindex(sdb.map)
@inline function Base.push!(sdb::SubDatabase, v)
    error("push! unsupported operation on SubDatabase")
end
@inline Base.eltype(sdb::SubDatabase) = eltype(sdb.db)

#
# Generic functions
#
function Base.iterate(db::AbstractDatabase, state::Int=1)
    n = length(db)
    if n == 0 || state > n
        nothing
    else
        @inbounds db[state], state + 1
    end
end
@inline Base.view(db::AbstractDatabase, map) = SubDatabase(db, map)
@inline Base.size(db::AbstractDatabase) = (length(db),)
@inline Random.rand(db::AbstractDatabase) = @inbounds db[rand(eachindex(db))]
@inline Random.rand(db::AbstractDatabase, n::Integer) = SubDatabase(db, rand(1:length(db), n))
@inline Random.rand(db::SubDatabase, n::Integer) = SubDatabase(db.db, rand(db.map, n))