# This file is a part of SimilaritySearch.jl

#####################################
#
# Wrapper for matrix-like containers
#

"""
    struct MatrixDatabase{M<:AbstractMatrix} <: AbstractDatabase

    MatrixDatabase(matrix::AbstractMatrix)

Wraps a matrix-like object `matrix` into a `MatrixDatabase`, i.e., each column of `matrix` is taken as one
object of the database. It is a static, fixed-size database (no `push_item!`/`append_items!` support);
use [`BlockMatrixDatabase`](@ref) or [`VectorDatabase`](@ref) when incremental growth is needed.
Please see [`AbstractDatabase`](@ref) for general usage.

# Examples

```julia
matrix = rand(Float32, 8, 100)  # 100 objects of dimension 8
db = MatrixDatabase(matrix)
db[1]        # the first object (a view of the first column)
length(db)   # 100
```
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

@inline Base.getindex(db::MatrixDatabase{<:DenseArray}, i::Integer) = view(db.matrix, :, i)
@inline Base.getindex(db::MatrixDatabase, i::Integer) = view(db.matrix, :, i)
@inline Base.setindex!(db::MatrixDatabase, value, i::Integer) = @inbounds (db.matrix[:, i] .= value)

"""
    push_item!(db::MatrixDatabase, v)

Not supported; `MatrixDatabase` is a fixed-size wrapper over a matrix. Use [`BlockMatrixDatabase`](@ref)
or [`VectorDatabase`](@ref) instead if you need to grow the database.
"""
@inline push_item!(db::MatrixDatabase, v) = error("push! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")

"""
    append_items!(a::MatrixDatabase, b)

Not supported; `MatrixDatabase` is a fixed-size wrapper over a matrix. Use [`BlockMatrixDatabase`](@ref)
or [`VectorDatabase`](@ref) instead if you need to grow the database.
"""
@inline append_items!(a::MatrixDatabase, b) = error("append! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.length(db::MatrixDatabase) = size(db.matrix, 2)


"""
    struct BlockMatrixDatabase{Dim,NumType,NumBits} <: AbstractDatabase

Stores objects of dimension `Dim` and element type `NumType` in a growable collection of dense matrix
blocks, each block holding `2^NumBits` columns/objects. It behaves like [`MatrixDatabase`](@ref) (each
column is one object, backed by contiguous matrices for fast access) but additionally supports
`push_item!`/`append_items!`, allocating a new block whenever the current one fills up. This makes it a
good fit when you need to incrementally append large numbers of items without paying the cost of
reallocating and copying a single growing matrix.

# Fields
- `blocks`: the list of dense matrix blocks
- `len`: current number of stored objects (a `Ref` so it can be mutated in place)

Please see [`AbstractDatabase`](@ref) for general usage.
"""
struct BlockMatrixDatabase{Dim,NumType,NumBits} <: AbstractDatabase
    blocks::Vector{Matrix{NumType}}  # array of matrices
    len::Ref{Int}
end

"""
    BlockMatrixDatabase(Dim::Int, ::Type{NumType}=Float32, NumBits::Int=8) where {NumType<:Number}

Creates an empty `BlockMatrixDatabase` for objects of dimension `Dim` and element type `NumType`, where
each internal block stores up to `2^NumBits` objects.
"""
function BlockMatrixDatabase(Dim::Int, ::Type{NumType}=Float32, NumBits::Int=8) where {NumType<:Number}
    BlockMatrixDatabase{Dim,NumType,NumBits}(Matrix{NumType}[], Ref(0))
end

"""
    BlockMatrixDatabase(M::AbstractMatrix, bitsize=8)

Creates a `BlockMatrixDatabase` from the columns of `M` (each column is one object), copying the data into
blocks of `2^bitsize` columns each. Unlike wrapping `M` directly with [`MatrixDatabase`](@ref), the result
supports further growth via `push_item!`/`append_items!`.

# Arguments
- `M`: the source matrix; `size(M, 1)` is taken as the object dimension
- `bitsize`: number of bits used to address positions within a block (block size is `2^bitsize`)

# Examples

```julia
matrix = rand(Float32, 8, 1000)
db = BlockMatrixDatabase(matrix)
push_item!(db, rand(Float32, 8))
length(db)  # 1001
```
"""
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
    @inbounds view(db.blocks[b], :, i)
end

@inline function Base.setindex!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, value, i::Integer) where {Dim,NumType,NumBits}
    b, i = _get_block_and_pos(NumBits, i)
    @inbounds db.blocks[b][:, i] .= value
end

"""
    push_item!(db::BlockMatrixDatabase, v::AbstractVector)

Appends `v` as a new object at the end of `db`, allocating a new internal block when the current one is full.
"""
@inline function push_item!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, v::AbstractVector) where {Dim,NumType,NumBits}
    n = db.len[] + 1
    b, i = _get_block_and_pos(NumBits, n)
    # @show b, i, n, Dim, NumType, NumBits, length(db), length(db.blocks), size(db.blocks[1])
    if i == 1
        M = Matrix{NumType}(undef, Dim, 1 << NumBits)
        @inbounds M[:, 1] .= v
        push!(db.blocks, M)
    else
        @inbounds db.blocks[b][:, i] .= v
    end

    db.len[] += 1
end

"""
    append_items!(db::BlockMatrixDatabase, B)

Appends every object in `B` (e.g., an iterator of vectors, such as `eachcol` of a matrix) to the end of `db`.
"""
@inline function append_items!(db::BlockMatrixDatabase, B)
    for b in B
        push_item!(db, b)
    end

    db
end

@inline Base.length(db::BlockMatrixDatabase) = db.len[]
