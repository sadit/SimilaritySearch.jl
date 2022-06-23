# This file is a part of SimilaritySearch.jl

#####################################
#
# Wrapper for matrix-like containers
#

"""
    struct MatrixDatabase{M<:AbstractDatabase} <: AbstractDatabase

    MatrixDatabase(matrix::AbstractMatrix)

Wraps a matrix-like object `matrix` into a `MatrixDatabase`. Please see [`AbstractDatabase`](@ref) for general usage.
"""
struct MatrixDatabase{M<:AbstractMatrix} <: AbstractDatabase
    matrix::M  # abstract matrix
end

@inline Base.eltype(db::MatrixDatabase) = AbstractVector{eltype(db.matrix)} 

"""
    MatrixDatabase(V::MatrixDatabase)    

Creates another `MatrixDatabase` from another `MatrixDatabase`. They will share their internal data. Please see [`AbstractDatabase`](@ref) for general usage.
"""
MatrixDatabase(V::MatrixDatabase) = MatrixDatabase(V.matrix)

@inline Base.getindex(db::MatrixDatabase{<:StrideArray}, i::Integer) = view(db.matrix, :, i)
@inline Base.getindex(db::MatrixDatabase{<:DenseArray}, i::Integer) = PtrArray(view(db.matrix, :, i))
@inline Base.getindex(db::MatrixDatabase, i::Integer) = view(db.matrix, :, i)
@inline Base.setindex!(db::MatrixDatabase, value, i::Integer) = @inbounds (db.matrix[:, i] .= value)
@inline Base.push!(db::MatrixDatabase, v) = error("push! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.append!(a::MatrixDatabase, b) = error("append! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.length(db::MatrixDatabase) = size(db.matrix, 2)
