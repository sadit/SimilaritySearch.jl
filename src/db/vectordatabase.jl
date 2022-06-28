# This file is a part of SimilaritySearch.jl

#
# Wrapper for array-like containers
#
"""
    struct VectorDatabase{V} <: AbstractDatabase

A wrapper for vector-like databases
"""
struct VectorDatabase{V} <: AbstractDatabase
    vecs::V  # abstract vector or something that looks like a vector
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

@inline Base.getindex(db::VectorDatabase, i::Integer) = db.vecs[i]
@inline Base.setindex!(db::VectorDatabase, value, i::Integer) = setindex!(db.vecs, value, i)
@inline Base.length(db::VectorDatabase) = length(db.vecs)
@inline Base.push!(db::VectorDatabase, v) = (push!(db.vecs, v); db)

@inline function Base.append!(db::VectorDatabase, B)
    for b in B
        push!(db, b)
    end

    db
end
