# This file is a part of SimilaritySearch.jl

#
# Wrapper for array-like containers
#
"""
    struct VectorDatabase{V} <: AbstractDatabase

Wraps a vector-like object `vecs` (e.g., a `Vector` of vectors, or any structure supporting `getindex`,
`setindex!`, `length`, `push!`) into an `AbstractDatabase`, i.e., each element of `vecs` is one object of
the database. Unlike [`MatrixDatabase`](@ref)/[`StrideMatrixDatabase`](@ref), it can hold objects of any
type (not just columns of a matrix) and supports growth via `push_item!`/`append_items!`.

# Fields
- `vecs`: the underlying vector-like container of objects

Please see [`AbstractDatabase`](@ref) for general usage.

# Examples

```julia
db = VectorDatabase([rand(Float32, 8) for _ in 1:100])  # 100 objects of dimension 8
db[1]         # the first object
length(db)    # 100

empty_db = VectorDatabase()  # an empty VectorDatabase{Vector{Float32}}
push_item!(empty_db, rand(Float32, 8))
```
"""
struct VectorDatabase{V} <: AbstractDatabase
    vecs::V  # abstract vector or something that looks like a vector
end

function show(io::IO, db::VectorDatabase; prefix="", indent="  ")
    println(io, prefix, "VectorDatabase:")
    prefix = prefix * indent
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "length: ", length(db))
end

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

Creates a `VectorDatabase` from another `VectorDatabase`. Both objects will share their internal data.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(V::VectorDatabase) = VectorDatabase(V.vecs)

"""
    VectorDatabase(; type=Vector{Float32})

Creates an empty `VectorDatabase` where each object is of type `type`.
Please see [`AbstractDatabase`](@ref) for general usage.
"""
VectorDatabase(; type=Vector{Float32}) = VectorDatabase(type[])

Base.Base.@propagate_inbounds @inline Base.getindex(db::VectorDatabase, i::Integer) = db.vecs[i]
@inline Base.setindex!(db::VectorDatabase, value, i::Integer) = setindex!(db.vecs, value, i)
@inline Base.length(db::VectorDatabase) = length(db.vecs)

"""
    push_item!(db::VectorDatabase, v)

Appends `v` as a new object at the end of `db`.
"""
@inline push_item!(db::VectorDatabase, v) = (push!(db.vecs, v); db)

"""
    append_items!(db::VectorDatabase, B)

Appends every object in `B` to the end of `db`.
"""
@inline function append_items!(db::VectorDatabase, B)
    for b in B
        push!(db.vecs, b)
    end

    db
end
