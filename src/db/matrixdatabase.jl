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

function show(io::IO, db::MatrixDatabase; prefix="", indent="  ")
    println(io, prefix, "MatrixDatabase:")
    prefix = prefix * indent
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "size: ", size(db.matrix))
end

@inline Base.eltype(db::MatrixDatabase) = typeof(db[1])

# @inline Base.getindex(db::MatrixDatabase{<:StrideArray}, i::Integer) = view(db.matrix, :, i)
@inline Base.getindex(db::MatrixDatabase{<:DenseArray}, i::Integer) = view(db.matrix, :, i)
#@inline Base.getindex(db::MatrixDatabase{Matrix{Float32}}, i::Integer) = PtrArray(view(db.matrix, :, i))
#@inline Base.getindex(db::MatrixDatabase{Matrix{Float64}}, i::Integer) = PtrArray(view(db.matrix, :, i))
@inline Base.getindex(db::MatrixDatabase, i::Integer) = view(db.matrix, :, i)
@inline Base.setindex!(db::MatrixDatabase, value, i::Integer) = @inbounds (db.matrix[:, i] .= value)
@inline push_item!(db::MatrixDatabase, v) = error("push! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline append_items!(a::MatrixDatabase, b) = error("append! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.length(db::MatrixDatabase) = size(db.matrix, 2)


"""
    struct StrideMatrixDatabase{M<:StrideArray} <: AbstractDatabase

    StrideMatrixDatabase(matrix::StrideArray)

Wraps a matrix object into a `StrideArray` and wrap it as a database. Please see [`AbstractDatabase`](@ref) for general usage.
"""
struct StrideMatrixDatabase{M<:StrideArray} <: AbstractDatabase
    matrix::M
end

function show(io::IO, db::StrideMatrixDatabase; prefix="", indent="  ")
    println(io, prefix, "StrideMatrixDatabase:")
    prefix = prefix * indent
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "size: ", size(db.matrix))
end

#StrideMatrixDatabase(M::Matrix) = StrideMatrixDatabase(StrideArray(M, StaticInt.(size(M))))
StrideMatrixDatabase(M::Matrix) = StrideMatrixDatabase(StrideArray(M, (size(M))))

@inline Base.eltype(db::StrideMatrixDatabase) = typeof(view(db.matrix, :, 1))

@inline Base.getindex(db::StrideMatrixDatabase, i::Integer) = view(db.matrix, :, i)
@inline Base.setindex!(db::StrideMatrixDatabase, value, i::Integer) = @inbounds (db.matrix[:, i] .= value)
@inline push_item!(db::StrideMatrixDatabase, v) = error("push! is not supported for StrideMatrixDatabase, please see DynamicMatrixDatabase")
@inline append_items!(a::StrideMatrixDatabase, b) = error("append! is not supported for StrideMatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.length(db::StrideMatrixDatabase) = size(db.matrix, 2)
