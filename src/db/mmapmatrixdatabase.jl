# This file is a part of SimilaritySearch.jl

#####################################
#
# Disk-backed, mmap'd matrix database
#

using Mmap

const MMAP_MATRIX_DB_MAGIC = 0x4D4D444231000001 % UInt64
const MMAP_MATRIX_DB_VERSION = Int32(1)
"""
Size (in bytes) of the fixed header at the start of a `MMapMatrixDatabase` file. It matches the
usual OS page size so that the data segment (which follows immediately) starts at a page boundary,
as required by `mmap` and convenient for efficient access.
"""
const MMAP_MATRIX_DB_HEADER_SIZE = 4096

const _MMAP_MATRIX_DB_MAGIC_OFFSET = 0
const _MMAP_MATRIX_DB_VERSION_OFFSET = 8
const _MMAP_MATRIX_DB_DIM_OFFSET = 12
const _MMAP_MATRIX_DB_NUMTYPE_OFFSET = 20
const _MMAP_MATRIX_DB_N_OFFSET = 24

const _MMAP_MATRIX_DB_NUMTYPE_CODES = Dict{DataType,Int32}(
    Float16 => 1, Float32 => 2, Float64 => 3,
    Int8 => 4, UInt8 => 5,
    Int16 => 6, UInt16 => 7,
    Int32 => 8, UInt32 => 9,
    Int64 => 10, UInt64 => 11,
)

const _MMAP_MATRIX_DB_NUMTYPE_DECODE = Dict{Int32,DataType}(v => k for (k, v) in _MMAP_MATRIX_DB_NUMTYPE_CODES)

_mmap_numtype_code(::Type{T}) where {T} =
    get(_MMAP_MATRIX_DB_NUMTYPE_CODES, T) do
        error("MMapMatrixDatabase: unsupported element type $T")
    end

_mmap_numtype_from_code(code::Integer) =
    get(_MMAP_MATRIX_DB_NUMTYPE_DECODE, Int32(code)) do
        error("MMapMatrixDatabase: unknown element type code $code found in file header")
    end

"""
    mutable struct MMapMatrixDatabase{Dim,NumType} <: AbstractDatabase

Stores objects of dimension `Dim` and element type `NumType` in a `Dim × capacity` matrix that is
memory-mapped onto a file, i.e., it behaves like [`BlockMatrixDatabase`](@ref) (grows with
`push_item!`/`append_items!`) but its storage lives on disk instead of in RAM. This makes it a good fit
for datasets that do not fit comfortably in memory, or that must survive process restarts.

The physical capacity of the file (in number of columns) is preallocated in doubling blocks -- as with
`BlockMatrixDatabase`'s `2^NumBits` blocks -- and only grows (extending and remapping the file) when
`push_item!`/`append_items!` need more room than is currently mapped; it is not remapped on every single
insertion. The *logical* length `n` (i.e. `length(db)`) is independent from the physical capacity and is
persisted in a small header at the start of the file, so it survives closing and reopening the database.

# Durability is opt-in: call `flush`

`push_item!`/`append_items!` update `n` in memory (so `length(db)` is correct right away within the
same process) and write the new columns into the mapped data, but do **not** msync those bytes or
persist/fsync the advanced `n` into the header -- that is exactly what [`flush`](@ref) does, and it
is the caller's responsibility to call it whenever *it* considers durability to matter (once per
batch, on a timer, before a deliberate checkpoint, ...), not something either mutating function does
on your behalf. `close`/the finalizer call `flush` once as a last-resort safety net, but garbage
collection timing is not a guarantee, so it is not a substitute for calling `flush` deliberately: a
process that pushes/appends and then crashes or is killed before an explicit `flush` (or a clean
`close`) loses everything added since the last `flush`. What `flush` still guarantees when you do
call it: `n` only ever advances on disk after the corresponding bytes are themselves durable, so a
crash *during* a `flush` never leaves the header pointing at data that isn't there.

Please see [`AbstractDatabase`](@ref) for general usage.

# Concurrency

Concurrent `push_item!`/`append_items!` calls from multiple threads on the *same* database are **not**
safe without external synchronization (e.g. a lock); they race on `n` and on the growth/remap logic.
[`flush`](@ref) is in the same category and for the same reason -- it reads `db.n`/`db.data`, both of
which a concurrent `push_item!`/`append_items!` mutates -- so calling it from a different thread than
the one doing the writing, without synchronization, is exactly as unsafe as two writers would be; it is
not a read-only operation just because it doesn't add an object.
A concurrent grow/remap (triggered by a writer) racing against a reader's `getindex` is safe in the sense
that it will not segfault -- the reader either sees the old, still-valid mapped array or the new one, since
old mappings are only released once nothing references them -- but it is still recommended to avoid mixing
growth and reads across threads without synchronization, since a `getindex` racing a `push_item!` is not
guaranteed to observe a consistent `n`/data pair.

# Examples

```julia
db = MMapMatrixDatabase("/tmp/mydb.mmapdb", 8, Float32)  # 8-dimensional Float32 objects
push_item!(db, rand(Float32, 8))
length(db)  # 1
close(db)

db2 = MMapMatrixDatabase("/tmp/mydb.mmapdb")  # reopens, restoring Dim/NumType/length from the header
length(db2)  # 1
close(db2)
```
"""
mutable struct MMapMatrixDatabase{Dim,NumType} <: AbstractDatabase
    path::String
    io::IOStream
    data::Matrix{NumType}
    n::Int
    read_only::Bool
    closed::Bool
end

function _mmap_matrix_db_write_header(io::IOStream, dim::Int, ::Type{NumType}, n::Int) where {NumType}
    seekstart(io)
    write(io, MMAP_MATRIX_DB_MAGIC)
    write(io, MMAP_MATRIX_DB_VERSION)
    write(io, Int64(dim))
    write(io, _mmap_numtype_code(NumType))
    write(io, Int64(n))
    flush(io)
    ccall(:fsync, Cint, (Cint,), fd(io))
    nothing
end

function _mmap_matrix_db_persist_n!(io::IOStream, n::Int)
    seek(io, _MMAP_MATRIX_DB_N_OFFSET)
    write(io, Int64(n))
    flush(io)
    ccall(:fsync, Cint, (Cint,), fd(io))
    nothing
end

"""
    MMapMatrixDatabase(path::AbstractString, dim::Integer, ::Type{NumType}=Float32; capacity_bits::Integer=8)

Creates a new `MMapMatrixDatabase` backed by a fresh file at `path`, for objects of dimension `dim` and
element type `NumType`. The file is preallocated for `2^capacity_bits` objects (256 by default) and doubles
its capacity (extending and remapping the file) whenever it fills up. Errors if `path` already exists; use
[`MMapMatrixDatabase(path)`](@ref) to reopen an existing file.
"""
function MMapMatrixDatabase(path::AbstractString, dim::Integer, ::Type{NumType}=Float32;
                             capacity_bits::Integer=8) where {NumType<:Number}
    isfile(path) && error("MMapMatrixDatabase: file `$path` already exists; use MMapMatrixDatabase(path) to reopen it")
    dim = Int(dim)
    capacity = 1 << Int(capacity_bits)
    io = open(path, "w+")
    data = try
        truncate(io, MMAP_MATRIX_DB_HEADER_SIZE + dim * capacity * sizeof(NumType))
        _mmap_matrix_db_write_header(io, dim, NumType, 0)
        Mmap.mmap(io, Matrix{NumType}, (dim, capacity), MMAP_MATRIX_DB_HEADER_SIZE; grow=false, shared=true)
    catch e
        close(io)
        rethrow(e)
    end

    db = MMapMatrixDatabase{dim,NumType}(String(path), io, data, 0, false, false)
    finalizer(_mmap_matrix_db_finalize!, db)
    db
end

"""
    MMapMatrixDatabase(path::AbstractString; read_only::Bool=false)

Reopens an existing `MMapMatrixDatabase` file at `path`, restoring its dimension, element type, and
logical length from the file's own header (they are not re-derived from external arguments). Pass
`read_only=true` to map the file without write permission, e.g. so a second process can inspect it while
another process is writing to it; in that mode `push_item!`/`append_items!`/`setindex!` error.
"""
function MMapMatrixDatabase(path::AbstractString; read_only::Bool=false)
    isfile(path) || error("MMapMatrixDatabase: file `$path` does not exist; use MMapMatrixDatabase(path, dim, NumType) to create one")
    io = open(path, read_only ? "r" : "r+")
    data = try
        magic = read(io, UInt64)
        magic == MMAP_MATRIX_DB_MAGIC || error("MMapMatrixDatabase: `$path` is not a valid MMapMatrixDatabase file (bad magic)")
        version = read(io, Int32)
        version == MMAP_MATRIX_DB_VERSION || error("MMapMatrixDatabase: unsupported format version $version in `$path`")
        dim = Int(read(io, Int64))
        NumType = _mmap_numtype_from_code(read(io, Int32))
        n = Int(read(io, Int64))
        databytes = filesize(path) - MMAP_MATRIX_DB_HEADER_SIZE
        capacity = databytes ÷ (dim * sizeof(NumType))
        capacity >= n || error("MMapMatrixDatabase: corrupt file `$path`: physical capacity ($capacity) is smaller than the recorded length ($n)")
        (dim, NumType, n, Mmap.mmap(io, Matrix{NumType}, (dim, capacity), MMAP_MATRIX_DB_HEADER_SIZE; grow=false, shared=true))
    catch e
        close(io)
        rethrow(e)
    end

    dim, NumType, n, mat = data
    db = MMapMatrixDatabase{dim,NumType}(String(path), io, mat, n, read_only, false)
    finalizer(_mmap_matrix_db_finalize!, db)
    db
end

"""
    flush(db::MMapMatrixDatabase)

Makes every object added so far durable: `msync`s the mapped data and persists/`fsync`s the
current logical length `n` into the file's header. Neither [`push_item!`](@ref) nor
[`append_items!`](@ref) do this on their own -- see the type docstring's "Durability" section.
A no-op on a `read_only` database (nothing to persist, and the underlying file isn't open for
writing in the first place).
"""
function Base.flush(db::MMapMatrixDatabase)
    db.read_only && return db
    Mmap.sync!(db.data)
    _mmap_matrix_db_persist_n!(db.io, db.n)
    db
end

function _mmap_matrix_db_finalize!(db::MMapMatrixDatabase)
    db.closed && return nothing
    db.closed = true
    flush(db)
    close(db.io)
    nothing
end

"""
    close(db::MMapMatrixDatabase)

Flushes (see [`flush`](@ref)), then unmaps and closes the underlying file. Idempotent --
calling it more than once (or letting the finalizer run afterwards) is safe.
"""
Base.close(db::MMapMatrixDatabase) = _mmap_matrix_db_finalize!(db)

function show(io::IO, db::MMapMatrixDatabase{Dim,NumType}; prefix="", indent="  ") where {Dim,NumType}
    println(io, prefix, "MMapMatrixDatabase{$Dim,$NumType}:")
    prefix = prefix * indent
    println(io, prefix, "path: ", db.path)
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "length: ", length(db))
    println(io, prefix, "capacity: ", size(db.data, 2))
    println(io, prefix, "read_only: ", db.read_only)
end

@inline Base.eltype(db::MMapMatrixDatabase) = typeof(db[1])
@inline Base.length(db::MMapMatrixDatabase) = db.n

@inline Base.getindex(db::MMapMatrixDatabase, i::Integer) = @inbounds view(db.data, :, i)

@inline function Base.setindex!(db::MMapMatrixDatabase, value, i::Integer)
    db.read_only && error("MMapMatrixDatabase: cannot setindex! on a read_only database")
    @inbounds db.data[:, i] .= value
end

function _mmap_matrix_db_grow!(db::MMapMatrixDatabase{Dim,NumType}, mincap::Int) where {Dim,NumType}
    newcap = max(1, size(db.data, 2))
    while newcap < mincap
        newcap *= 2
    end
    db.data = Mmap.mmap(db.io, Matrix{NumType}, (Dim, newcap), MMAP_MATRIX_DB_HEADER_SIZE; grow=true, shared=true)
    db.data
end

"""
    push_item!(db::MMapMatrixDatabase, v::AbstractVector)

Appends `v` as a new object at the end of `db`, growing (extending and remapping) the underlying
file when the current capacity is exceeded. `length(db)` reflects `v` immediately, but nothing is
made durable by this call -- see the type docstring's "Durability" section, and call
[`flush`](@ref) when that matters to you.
"""
function push_item!(db::MMapMatrixDatabase{Dim,NumType}, v::AbstractVector) where {Dim,NumType}
    db.read_only && error("MMapMatrixDatabase: cannot push_item! on a read_only database")
    n = db.n + 1
    n > size(db.data, 2) && _mmap_matrix_db_grow!(db, n)
    @inbounds db.data[:, n] .= v
    db.n = n
    db
end

"""
    append_items!(db::MMapMatrixDatabase, B)

Appends every object in `B` (e.g., an iterator of vectors, such as `eachcol` of a matrix) to the end
of `db`, growing the underlying file as needed. `length(db)` reflects every item of `B` immediately,
but as with [`push_item!`](@ref), nothing is made durable by this call -- call [`flush`](@ref) when
that matters to you.
"""
function append_items!(db::MMapMatrixDatabase{Dim,NumType}, B) where {Dim,NumType}
    db.read_only && error("MMapMatrixDatabase: cannot append_items! on a read_only database")
    n = db.n
    for v in B
        n += 1
        n > size(db.data, 2) && _mmap_matrix_db_grow!(db, n)
        @inbounds db.data[:, n] .= v
    end

    db.n = n
    db
end
