# This file is a part of SimilaritySearch.jl

# Database interface
using Random

export AbstractDatabase, MatrixDatabase, DynamicMatrixDatabase, VectorDatabase, SubDatabase

"""
    abstract type AbstractDatabase end

Base type to represent databases. A database is a collection of objects that can be accessed like a
similar interface to `AbstractVector`. It is separated to allow `SimilaritySearch` methods to know what is a database and what is an object (since most object
representations will look as vectors and matrices). 

The basic implementations are:
- `MatrixDatabase`: A wrapper for object-vectors stored in a `Matrix`, columns are the objects. It is static.
- `DynamicMatrixDatabase`: A dynamic representation for vectors that allows adding new vectors.
- `VectorDatabase`: A wrapper for vector-like structures. It can contain any kind of objects.
- `SubDatabase`: A sample of a given database

In particular, the storage details are not used by `VectorDatabase` and `MatrixDatabase`.
For instance, it is possible to use matrices like `Matrix`, `SMatrix` or `StrideArrays`; or even use
generated objects with `VectorDatabase` (supporting a vector-like interface).

If the storage backend support it, it is possible to use vector operations, for example:
- get the `i`-th element `obj = db[i]`, elements in the database are identified by position
- get the elements list in a list of indices `lst` as `db[lst]` (also using `view`)
- set a value at the `i`-th element `db[i] = obj`
- random sampling `rand(db)`, `rand(db, 3)`
- iterate and collect objects in the database
- get the number of elements in the database `length(db)`
- add new objects to the end of the database (not all internal containers will support it)
  - `push!(db, u)` adds a single element `u`
  - `append!(db, lst)` adds a list of objects to the end of the database
"""
abstract type AbstractDatabase end


"""
    view(db::AbstractDatabase, map)

Constructs a `SubDatabase` from `db` using the specified indexes in `map` (e.g., an array or an slice of indices)
"""
@inline Base.view(db::AbstractDatabase, map) = SubDatabase(db, map)
"""
    size(db::AbstractDatabase)

The size of the database `(length(db),)`. There is no concept of explicit dimension for `AbstractDatabase` implementations.
"""
@inline Base.size(db::AbstractDatabase) = (length(db),)

"""
    rand(db::AbstractDatabase)

Retrieves a random element from `db`
"""
@inline Random.rand(db::AbstractDatabase) = @inbounds db[rand(eachindex(db))]

"""
    rand(db::AbstractDatabase, n)

Retrieves `n` random elements from `db`, returnes a `SubDatabase` object
"""
@inline Random.rand(db::AbstractDatabase, n::Integer) = SubDatabase(db, rand(1:length(db), n))

"""
    firstindex(db::AbstractDatabase)

First index in `db` (1)
"""
@inline Base.firstindex(db::AbstractDatabase) = 1

"""
    lastindex(db::AbstractDatabase)

Last index in `db` (1)
"""
@inline Base.lastindex(db::AbstractDatabase) = length(db)

"""
    eachindex(db::AbstractDatabase)

An index iterator of `db`
"""
@inline Base.eachindex(db::AbstractDatabase) = firstindex(db):lastindex(db)

"""
    getindex(db::AbstractDatabase, lst)

A subset of `db` (using a collection of indexes `lst`)
"""
@inline Base.getindex(db::AbstractDatabase, lst::AbstractVector{<:Integer}) = SubDatabase(db, lst)

"""
    struct DynamicMatrixDatabase{DType,Dim} <: AbstractDatabase

A dynamic matrix with elements of type `DType` and dimension `Dim` 
"""
struct DynamicMatrixDatabase{DType,Dim} <: AbstractDatabase
    data::Vector{DType}
end

"""
    struct MatrixDatabase{M<:AbstractDatabase} <: AbstractDatabase

    MatrixDatabase(matrix::AbstractMatrix)

Wraps a matrix-like object `matrix` into a `MatrixDatabase`. Please see [`AbstractDatabase`](@ref) for general usage.
"""
struct MatrixDatabase{M<:AbstractMatrix} <: AbstractDatabase
    matrix::M  # abstract matrix
end

"""
    struct VectorDatabase{V} <: AbstractDatabase

A wrapper for vector-like databases
"""
struct VectorDatabase{V} <: AbstractDatabase
    vecs::V  # abstract vector or something that looks like a vector
end

#
# Dynamic matrix-like dataset, i.e., columns are objects. Stored as a large vector to support appending items
#

"""
    eltype(db::DynamicMatrixDatabase{DType,Dim})

The type of stored elements `AbstractVector{DType}`
"""
@inline Base.eltype(db::DynamicMatrixDatabase{DType,Dim}) where {DType,Dim} = AbstractVector{DType}

"""
    DynamicMatrixDatabase(matrix::AbstractMatrix{DType})

Creates a `DynamicMatrixDatabase` from a matrix-like object. Please see [`AbstractDatabase`](@ref) for general usage.
"""
DynamicMatrixDatabase(matrix::AbstractMatrix{DType}) where DType = DynamicMatrixDatabase{DType,size(matrix, 1)}(vec(matrix))

"""
    DynamicMatrixDatabase(::Type{DType}, Dim::Integer)

Creates an empty `DynamicMatrixDatabase` such that `length(db[i]) == Dim` and `eltype(db[i]) == DType`
"""
DynamicMatrixDatabase(::Type{DType}, Dim::Integer) where DType = DynamicMatrixDatabase{DType,Dim}(Vector{DType}(undef, 0))

"""
    DynamicMatrixDatabase(V::DynamicMatrixDatabase{DType,Dim})    

Creates a `DynamicMatrixDatabase` from another `DynamicMatrixDatabase`, they will share their internal data. Please see [`AbstractDatabase`](@ref) for general usage.
"""
DynamicMatrixDatabase(V::DynamicMatrixDatabase{DType,Dim}) where {DType,Dim} = DynamicMatrixDatabase{DType,Dim}(V.data)

"""
    DynamicMatrixDatabase(M::MatrixDatabase)

Creates a `DynamicMatrixDatabase` from a `MatrixDatabase`, copies internal data. Please see [`AbstractDatabase`](@ref) for general usage.
"""
DynamicMatrixDatabase(M::MatrixDatabase) = DynamicMatrixDatabase{eltype{M.data},size(M.matrix, 1)}(copy(vec(M.matrix)))

@inline function Base.getindex(db::DynamicMatrixDatabase{DType,Dim}, i::Integer) where {DType,Dim}
    ep = i * Dim
    @inbounds @view db.data[(ep - Dim + 1):ep]
end
@inline Base.setindex!(db::DynamicMatrixDatabase, value, i) = @inbounds (db.data[i] .= value)

@inline Base.length(db::DynamicMatrixDatabase{DType,Dim}) where {DType,Dim} = length(db.data) ÷ Dim
@inline function Base.push!(db::DynamicMatrixDatabase{DType,Dim}, v) where {DType,Dim}
    append!(db.data, v)
    db
end

@inline Base.append!(a::DynamicMatrixDatabase{DType,Dim}, b::DynamicMatrixDatabase{DType2,Dim})  where {DType,Dim,DType2} = append!(a.data, b.data)
@inline function Base.append!(db::DynamicMatrixDatabase{DType,Dim}, B) where {DType,Dim}
    sizehint!(db.data, length(db.data) + Dim * length(B))
    for b in B
        push!(db, b)
    end

    db
end

#
# Wrapper for matrix-like containers
#

@inline Base.eltype(db::MatrixDatabase) = AbstractVector{eltype(db.matrix)}


"""
    MatrixDatabase(V::MatrixDatabase)    

Creates another `MatrixDatabase` from another `MatrixDatabase`. They will share their internal data. Please see [`AbstractDatabase`](@ref) for general usage.
"""
MatrixDatabase(V::MatrixDatabase) = MatrixDatabase(V.matrix)

@inline Base.getindex(db::MatrixDatabase, i::Integer) = view(db.matrix, :, i)
@inline Base.setindex!(db::MatrixDatabase, value, i::Integer) = @inbounds (db.matrix[:, i] .= value)
@inline Base.push!(db::MatrixDatabase, v) = error("push! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.append!(a::MatrixDatabase, b) = error("append! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.length(db::MatrixDatabase) = size(db.matrix, 2)

#
# Wrapper for array-like containers
#

@inline Base.eltype(db::VectorDatabase) = eltype(db.vecs)

"""
    VectorDatabase(vecs::T)

Creates a `VectorDatabase` from `vecs`. Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(vecs::T) where {T<:AbstractVector} = VectorDatabase{T}(vecs)

"""
    VectorDatabase(M::AbstractMatrix)

Creates a `VectorDatabase` from a matrix-like object. It will copy columns as objects  of the new `VectorDatabase`.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(M::T) where {T<:AbstractMatrix} = VectorDatabase([Vector(c) for c in eachcol(M)])

"""
    VectorDatabase(D::AbstractDatabase)

Creates a `VectorDatabase` from an `AbstractDatabase`. It copies internal data.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(D::AbstractDatabase) = VectorDatabase([Vector(c) for c in D])

"""
    VectorDatabase(V::VectorDatabase)

Creates a `VectorDatabase` from another an `AbstractDatabase`. They will share their internal data.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(V::VectorDatabase) = VectorDatabase(V.vecs)

"""
    VectorDatabase(; type=Vector{Float32})

Creates an empty `VectorDatabase` where each object is of type `type`.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(; type=Vector{Float32}) = VectorDatabase(type[])

@inline Base.getindex(db::VectorDatabase, i::Integer) = @inbounds db.vecs[i]
@inline Base.setindex!(db::VectorDatabase, value, i::Integer) = @inbounds (db.vecs[i] = value)
@inline Base.length(db::VectorDatabase) = length(db.vecs)
@inline Base.push!(db::VectorDatabase, v) = (push!(db.vecs, v); db)

@inline function Base.append!(db::VectorDatabase, B)
    for b in B
        push!(db, b)
    end

    db
end


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

"""
    convert(::Type{AbstractDatabase}, M::AbstractDatabase)
    convert(::Type{AbstractDatabase}, M::AbstractMatrix)
    convert(::Type{AbstractDatabase}, M::Vector)
    convert(::Type{AbstractDatabase}, M::Vector{Any})
    convert(::Type{AbstractDatabase}, M::AbstractVector)
    convert(::Type{<:AbstractVector}, M::VectorDatabase)
    convert(::Type{<:AbstractVector}, M::AbstractDatabase)

Convenience functions to convert different kinds of data into a some kind of database. 
"""
Base.convert(::Type{AbstractDatabase}, M::AbstractDatabase) = M
Base.convert(::Type{AbstractDatabase}, M::AbstractMatrix) = MatrixDatabase(M)
Base.convert(::Type{AbstractDatabase}, M::Vector) = VectorDatabase(M)
Base.convert(::Type{AbstractDatabase}, M::Vector{Any}) = VectorDatabase(typeof(first(M)).(M))
Base.convert(::Type{AbstractDatabase}, M::AbstractVector) = VectorDatabase(typeof(first(M)).(M))
Base.convert(::Type{<:AbstractVector}, M::VectorDatabase{T}) where T = M.data
Base.convert(::Type{<:AbstractVector}, M::AbstractDatabase) = collect(M)

