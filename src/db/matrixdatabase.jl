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


"""
    struct BlockMatrixDatabase{M<:AbstractDatabase} <: AbstractDatabase

    BlockMatrixDatabase(bsize::Int)

"""
struct BlockMatrixDatabase{Dim,NumType,NumBits} <: AbstractDatabase
    blocks::Vector{Matrix{NumType}}  # array of matrices
    len::Ref{Int}
end

function BlockMatrixDatabase(Dim::Int, ::Type{NumType}=Float32, NumBits::Int=8) where {NumType<:Number}
    BlockMatrixDatabase{Dim,NumType,NumBits}(Matrix{NumType}[], Ref(0))
end

function BlockMatrixDatabase(M::AbstractMatrix, bitsize=8)
    dim = size(M, 1)
    B = BlockMatrixDatabase(dim, eltype(M), bitsize)
    append_items!(B, eachcol(M))
    B
end

function show(io::IO, db::BlockMatrixDatabase{Dim,NumType,NumBits}; prefix="", indent="  ") where {Dim,NumType,NumBits}
    println(io, prefix, "BlockMatrixDatabase{$Dim,$NumType,$NumBits}:")
    prefix = prefix * indent
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "size: ", (Dim, length(db)))
end

@inline Base.eltype(db::BlockMatrixDatabase) = typeof(db[1])

@inline function _get_block_and_pos(NumBits, i)
    mask = (1 << NumBits) - 1
    i -= 1
    b = (i >> NumBits) + 1
    pos = (i & mask) + 1
    b, pos
end

@inline function Base.getindex(db::BlockMatrixDatabase{Dim,NumType,NumBits}, i::Integer) where {Dim,NumType,NumBits}
    b, i = _get_block_and_pos(NumBits, i)
    view(db.blocks[b], :, i)
end

@inline function Base.setindex!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, value, i::Integer) where {Dim,NumType,NumBits}
    b, i = _get_block_and_pos(NumBits, i)
    db.blocks[b][:, i] .= value
end

@inline function push_item!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, v::AbstractVector) where {Dim,NumType,NumBits}
    n = db.len[] + 1
    b, i = _get_block_and_pos(NumBits, n)
    # @show b, i, n, Dim, NumType, NumBits, length(db), length(db.blocks), size(db.blocks[1])
    if i == 1
        M = Matrix{NumType}(undef, Dim, 1 << NumBits)
        M[:, 1] .= v
        push!(db.blocks, M)
    else
        db.blocks[b][:, i] .= v
    end

    db.len[] += 1
end

@inline function append_items!(db::BlockMatrixDatabase, B)
    for b in B
        push_item!(db, b)
    end

    db
end

@inline Base.length(db::BlockMatrixDatabase) = db.len[]
