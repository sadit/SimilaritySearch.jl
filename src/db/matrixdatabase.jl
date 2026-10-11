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
use [`BlockMatrixDatabase`](@ref) or [`VectorDatabase`](@ref SimilaritySearch.VectorDatabase) when incremental growth is needed.
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
or [`VectorDatabase`](@ref SimilaritySearch.VectorDatabase) instead if you need to grow the database.
"""
@inline push_item!(db::MatrixDatabase, v) = error("push! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")

"""
    append_items!(a::MatrixDatabase, b)

Not supported; `MatrixDatabase` is a fixed-size wrapper over a matrix. Use [`BlockMatrixDatabase`](@ref)
or [`VectorDatabase`](@ref SimilaritySearch.VectorDatabase) instead if you need to grow the database.
"""
@inline append_items!(a::MatrixDatabase, b) = error("append! is not supported for MatrixDatabase, please see DynamicMatrixDatabase")
@inline Base.length(db::MatrixDatabase) = size(db.matrix, 2)


"""
    struct BlockMatrixDatabase{Dim,NumType,NumBits} <: AbstractDatabase

Stores objects of dimension `Dim` and element type `NumType` in a growable collection of dense matrix
blocks, each block holding `2^NumBits` columns/objects (the last one may hold fewer columns, and grows
as items arrive). It behaves like [`MatrixDatabase`](@ref) (each
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
    defaultblockbits(Dim, NumType) -> Int

The default `NumBits` of a [`BlockMatrixDatabase`](@ref): blocks of at least 32 MB, so that at least
15 of every 16 of their bytes sit on 2 MB huge pages whatever the block's alignment (an 8 MB block
measured 66%), or 256 columns when [`page_spread`](@ref) is off (the layout up to 1.6.4). The last
block holds only what it needs, so a small database does not pay for the size.
"""
function defaultblockbits(Dim::Integer, ::Type{NumType}) where {NumType}
    PAGE_SPREAD[] || return 8
    max(8, ceil(Int, log2(cld(1 << 25, max(1, Dim * sizeof(NumType))))))
end

"""
    BlockMatrixDatabase(Dim::Int, ::Type{NumType}=Float32, NumBits::Int=defaultblockbits(Dim, NumType)) where {NumType<:Number}

Creates an empty `BlockMatrixDatabase` for objects of dimension `Dim` and element type `NumType`, where
each internal block stores up to `2^NumBits` objects (by default, blocks of at least 32 MB; see
[`defaultblockbits`](@ref)).
"""
function BlockMatrixDatabase(Dim::Int, ::Type{NumType}=Float32, NumBits::Int=defaultblockbits(Dim, NumType)) where {NumType<:Number}
    BlockMatrixDatabase{Dim,NumType,NumBits}(Matrix{NumType}[], Ref(0))
end

"""
    BlockMatrixDatabase(M::AbstractMatrix, bitsize=defaultblockbits(size(M, 1), eltype(M)))

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
function BlockMatrixDatabase(M::AbstractMatrix, bitsize=defaultblockbits(size(M, 1), eltype(M)))
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

# Room for column `need` of block `b`. One push at a time (`exact=false`): a new block starts at 256
# columns and the last one doubles, up to 2^NumBits. A bulk append (`exact=true`) asks for exactly what
# it fills. Either way a database that stopped growing holds little unused room.
function _blockroom!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, b::Int, need::Int, exact::Bool=false) where {Dim,NumType,NumBits}
    cap = 1 << NumBits
    if b > length(db.blocks)
        push!(db.blocks, Matrix{NumType}(undef, Dim, min(cap, exact ? need : max(need, 256))))
    else
        M = db.blocks[b]
        if need > size(M, 2)
            M2 = Matrix{NumType}(undef, Dim, min(cap, exact ? need : max(need, 2size(M, 2))))
            copyto!(M2, 1, M, 1, length(M))
            db.blocks[b] = M2
        end
    end
    db
end

"""
    push_item!(db::BlockMatrixDatabase, v::AbstractVector)

Appends `v` as a new object at the end of `db`, making room in the last block or allocating a new one.
"""
@inline function push_item!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, v::AbstractVector) where {Dim,NumType,NumBits}
    n = db.len[] + 1
    b, i = _get_block_and_pos(NumBits, n)
    (b > length(db.blocks) || i > size(@inbounds(db.blocks[b]), 2)) && _blockroom!(db, b, i)
    @inbounds db.blocks[b][:, i] .= v
    db.len[] = n
end

# Room for `m` more items: the last block grows to what it needs, new blocks are full except the last,
# which gets exactly the columns left. Nothing is written: the pages are touched by whoever fills them.
function _reserve!(db::BlockMatrixDatabase{Dim,NumType,NumBits}, m::Integer) where {Dim,NumType,NumBits}
    n = db.len[] + m
    n == db.len[] && return db
    blast, ilast = _get_block_and_pos(NumBits, n)
    cap = 1 << NumBits
    for b in max(1, length(db.blocks)):blast
        _blockroom!(db, b, b == blast ? ilast : cap, true)
    end
    db
end

"""
    _appendby!(db, m, item)

Appends `item(1)`, ..., `item(m)` to `db`. A store that supports it reserves the room first and
fills it from all threads (see [`set_page_spread!`](@ref)), so `item` may run concurrently; the
default pushes them one by one.
"""
_appendby!(db::AbstractDatabase, m::Integer, item::F) where {F} = _pushall!(db, m, item)

function _pushall!(db::AbstractDatabase, m::Integer, item::F) where {F}
    for k in 1:m
        push_item!(db, item(k))
    end
    db
end

function _appendby!(db::BlockMatrixDatabase, m::Integer, item::F) where {F}
    _spreads(m) || return _pushall!(db, m, item)
    n0 = db.len[]
    _reserve!(db, m)
    db.len[] = n0 + m
    try
        _spread(m) do r
            for k in r
                db[n0 + k] = item(k)
            end
        end
    catch
        db.len[] = n0
        rethrow()
    end
    db
end

"""
    append_items!(db::BlockMatrixDatabase, B)

Appends every object in `B` (e.g., an iterator of vectors, such as `eachcol` of a matrix) to the end of
`db`. An indexable `B` (a vector, `eachcol`, a database) with many items is copied by all threads (see
[`set_page_spread!`](@ref)).
"""
function append_items!(db::BlockMatrixDatabase, B)
    if B isa Union{AbstractVector,AbstractDatabase}
        _appendby!(db, length(B), k -> B[k])
    else
        for b in B
            push_item!(db, b)
        end
    end

    db
end

@inline Base.length(db::BlockMatrixDatabase) = db.len[]
