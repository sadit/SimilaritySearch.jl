# This file is a part of SimilaritySearch.jl

# Database interface
using StrideArraysCore
using Random

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
  - `push_item!(db, u)` adds a single element `u`
  - `append_items!(db, lst)` adds a list of objects to the end of the database
"""
abstract type AbstractDatabase end


function show(io::IO, db::AbstractDatabase; prefix="", indent="  ")
    println(io, prefix, "AbstractDatabase:")
    prefix = prefix * indent
    println(io, prefix, "type: ", typeof(db))
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "length: ", length(db))
end

include("dynamicmatrixdatabase.jl")
include("matrixdatabase.jl")
include("vectordatabase.jl")
include("subdatabase.jl")

export AbstractDatabase, MatrixDatabase, StrideMatrixDatabase, DynamicMatrixDatabase, VectorDatabase, SubDatabase

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
@inline Base.getindex(db::AbstractDatabase, lst::AbstractVector) = SubDatabase(db, lst)
@inline Base.getindex(db::AbstractDatabase, lst::AbstractVector{Bool}) = SubDatabase(db, findall(lst))
@inline Base.getindex(db::AbstractDatabase, lst::BitArray) = SubDatabase(db, findall(lst))

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
Base.convert(::Type{<:AbstractVector}, M::VectorDatabase{T}) where {T} = M.data
Base.convert(::Type{<:AbstractVector}, M::AbstractDatabase) = collect(M)

"""
    MatrixDatabase(V)
    MatrixDatabase(V)

Creates another `MatrixDatabase` from another `AbstractDatabase`

Note that if `V` is a `MatrixDatabase` or a `StrideMatrixDatabase` then both objects will share their internal data.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
MatrixDatabase(V::MatrixDatabase) = MatrixDatabase(V.matrix)
MatrixDatabase(V::StrideMatrixDatabase) = MatrixDatabase(V.matrix)
function MatrixDatabase(V::AbstractDatabase)
    @assert length(V) > 0 "copy empty datasets is not allowed"
    MatrixDatabase(hcat(V...))
end
#MatrixDatabase(V::AbstractDatabase) = MatrixDatabase(hcat(V...))
StrideMatrixDatabase(V::MatrixDatabase) = StrideMatrixDatabase(V.matrix)
StrideMatrixDatabase(V::StrideMatrixDatabase) = StrideMatrixDatabase(V.matrix)
StrideMatrixDatabase(V::AbstractDatabase) = StrideMatrixDatabase(hcat(V...))

"""
    DynamicMatrixDatabase(M::MatrixDatabase)

Creates a `DynamicMatrixDatabase` from a `MatrixDatabase`, copies internal data.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
DynamicMatrixDatabase(M::MatrixDatabase) = DynamicMatrixDatabase{eltype{M.matrix},size(M.matrix, 1)}(copy(vec(M.matrix)))
DynamicMatrixDatabase(M::StrideMatrixDatabase) = DynamicMatrixDatabase{eltype{M.matrix},size(M.matrix, 1)}(copy(vec(M.matrix)))
