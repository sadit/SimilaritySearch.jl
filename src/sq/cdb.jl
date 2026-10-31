# This file is a part of SimilaritySearch.jl
#
# Two layouts of a quantized database besides `QuantDatabase`'s codes-plus-two-arrays: the codes
# alone (`GlobalCodesDatabase`), and the codes with their sums at the end of each row
# (`PackedQuantDatabase`). The three hold the same codes and give the same distances; they differ
# in what an evaluation has to bring from memory.

export GlobalCodesDatabase, PackedQuantDatabase

"""
    GlobalCodesDatabase{B,DB<:AbstractDatabase} <: AbstractDatabase

    GlobalCodesDatabase(db::GlobalQuantDatabase)
    GlobalCodesDatabase(bits::Integer, X::AbstractMatrix; minmax=nothing, storage=MatrixDatabase, quant=nothing, samplesize=0)
    GlobalCodesDatabase(bits::Integer, Q, minmax; dim=nothing)

A database of globally quantized vectors that keeps only the codes and the shared `min`/scale
pair: indexing yields an [`SQCodes`](@ref), the vector without its code sums. It is the layout
for `SqL2`, `L2` and `L1` between codes, the symmetric graph over the global family: between two
codes of the same scale these distances are `c²·Σ(a-b)²` and `c·Σ|a-b|` over the codes, and never
read `Sa`/`Saa`, which a [`GlobalQuantDatabase`](@ref) loads anyway, from two other arrays, for
every item it hands out.

Measured on a SearchGraph of 8-bit codes (SISAP 2025 ccnews and gooaq, 384 dimensions, QR
rotation, 64 threads, tuned for recall 0.9): the same graph, visits and answers, with the
database's `getindex` building the `SQVec` with zero sums instead of loading them, gave 11-14%
more queries per second (311 -> 276 ns·thread per evaluation on ccnews, 346 -> 312 on gooaq).
Prefetching the codes alone while still loading the sums lost 3-10%: the loads are what costs.
It also saves 8 bytes per vector.

Everything else still works, at a price: `NormCosine`, `Cosine`, codes of other scales and a
prepared [`SQQuery`](@ref) need the sums, and get them from one more pass over the codes per
evaluation. Use a `GlobalQuantDatabase`, or a [`PackedQuantDatabase`](@ref), for those, and for an
`AsymmetricSearchGraph` whose queries are `SQQuery`s.

The first constructor drops the sums of `db` (sharing its codes), the second quantizes `X` as
[`GlobalQuantDatabase`](@ref) does, and the third takes codes back with the range they were made
with. It grows like a `GlobalQuantDatabase` when its storage does.
"""
struct GlobalCodesDatabase{B,DB<:AbstractDatabase} <: AbstractDatabase
    E::SQMinC
    Q::DB
    dim::Int

    function GlobalCodesDatabase{B,DB}(E::SQMinC, Q::DB, dim::Integer) where {B,DB<:AbstractDatabase}
        B in (2, 4, 8) || throw(ArgumentError("GlobalCodesDatabase: B=$B must be 2, 4 or 8"))
        cpb = codesperbyte(Val(B))
        dim % cpb == 0 ||
            throw(ArgumentError("GlobalCodesDatabase: dim=$dim must be a multiple of $cpb ($cpb coordinates are packed per UInt8 at $B bits); pad the vectors"))
        new{B,DB}(E, Q, Int(dim))
    end
end

function GlobalCodesDatabase{B}(E::SQMinC, Q::Union{AbstractMatrix{UInt8},AbstractDatabase}; dim=nothing) where {B}
    Qdb = _asdb(Q)
    d = dim === nothing ? codesperbyte(Val(B)) * _rowbytes(Qdb) : Int(dim)
    GlobalCodesDatabase{B,typeof(Qdb)}(E, Qdb, d)
end

GlobalCodesDatabase(db::QuantDatabase{B,SQMinC}) where {B} = GlobalCodesDatabase{B,typeof(db.Q)}(db.E, db.Q, db.dim)
GlobalCodesDatabase(bits::Integer, X::AbstractMatrix; kwargs...) = GlobalCodesDatabase(GlobalQuantDatabase(bits, X; kwargs...))
GlobalCodesDatabase(bits::Integer, Q::Union{AbstractMatrix{UInt8},AbstractDatabase}, minmax; dim=nothing) =
    GlobalCodesDatabase{_checkbits(bits)}(_globalparams(Val(_checkbits(bits)), minmax), Q; dim)

codewidth(::GlobalCodesDatabase{B}) where {B} = B
isglobal(::GlobalCodesDatabase) = true
quantizer(::GlobalCodesDatabase{B}) where {B} = _quantmodule(B, true)

Base.length(db::GlobalCodesDatabase) = length(db.Q)
Base.eltype(db::GlobalCodesDatabase{B}) where {B} = length(db) > 0 ? typeof(db[1]) : SQCodes{B}
Base.@propagate_inbounds Base.getindex(db::GlobalCodesDatabase{B}, i::Integer) where {B} = SQCodes{B}(db.E, db.Q[i])

function show(io::IO, db::GlobalCodesDatabase{B}; prefix="", indent="  ") where {B}
    println(io, prefix, "GlobalCodesDatabase{$B}:")
    prefix = prefix * indent
    println(io, prefix, "dim: ", db.dim)
    println(io, prefix, "length: ", length(db))
    println(io, prefix, "storage: ", typeof(db.Q))
end

function quantize(db::GlobalCodesDatabase{B}, v::AbstractVector) where {B}
    _checkdim(db, v)
    codes = Vector{UInt8}(undef, db.dim ÷ codesperbyte(Val(B)))
    packcodes!(Val(B), codes, v, db.E.min, _globalscale(db.E))
    SQCodes{B}(db.E, codes)
end

# what the database stores for an item: a plain vector is quantized, a quantized one is checked
_asstored(db::GlobalCodesDatabase, v::AbstractVector) = quantize(db, v)
function _asstored(db::GlobalCodesDatabase{B}, v::_SQ{B}) where {B}
    _checkdim(db, v)
    v.E == db.E ||
        throw(ArgumentError("push_item!: the vector was quantized with other parameters than the database's ($(v.E) against $(db.E))"))
    v
end

push_item!(db::GlobalCodesDatabase, v) = (push_item!(db.Q, _asstored(db, v).V); db)

function _appendby!(db::GlobalCodesDatabase, m::Integer, item::F) where {F}
    _appendby!(db.Q, m, k -> _asstored(db, item(k)).V)
    db
end

function append_items!(db::GlobalCodesDatabase, items)
    items isa Union{AbstractVector,AbstractDatabase} && return _appendby!(db, length(items), k -> items[k])
    for v in items
        push_item!(db, v)
    end

    db
end

append_items!(db::GlobalCodesDatabase, X::AbstractMatrix) = append_items!(db, eachcol(X))

function Base.:(==)(a::GlobalCodesDatabase{B}, b::GlobalCodesDatabase{B}) where {B}
    length(a) == length(b) && a.dim == b.dim && a.E == b.E || return false
    all(i -> a.Q[i] == b.Q[i], eachindex(a))
end

@inline prefetchable(db::GlobalCodesDatabase) = prefetchable(db.Q)
@inline prefetch_item(db::GlobalCodesDatabase, i::Integer) = prefetch_item(db.Q, i)

function spreadcopy(db::GlobalCodesDatabase{B}) where {B}
    Q = spreadcopy(db.Q)
    GlobalCodesDatabase{B,typeof(Q)}(db.E, Q, db.dim)
end

"""
    PackedQuantDatabase{B,P,DB<:AbstractDatabase} <: AbstractDatabase

    PackedQuantDatabase(db::QuantDatabase; storage=MatrixDatabase)
    PackedQuantDatabase{B}(E, Q; dim)

A [`QuantDatabase`](@ref) whose per-vector quantities live in the row of each vector, after its
codes, instead of in arrays of their own: `Sa` and `Saa` as two `Float32`s in the global family
(`P == SQMinC`, the shared pair in `E`), and also that vector's `min` and scale in the per-vector
one (`P == Nothing`, `E === nothing`). The codes are padded to a multiple of 4 bytes first, so a
row of 384 8-bit codes takes 392 bytes in the global family and 400 in the per-vector one.
Indexing yields an [`SQVec`](@ref) with the same codes and sums a `QuantDatabase` gives, so every
distance returns the same value.

It is the layout for the distances that read the sums -- `NormCosine` and `Cosine` between codes,
`SqL2`/`L2` against a prepared [`SQQuery`](@ref) (the `AsymmetricSearchGraph`), the per-vector
family -- where a `QuantDatabase` brings each child from three or four places in memory and this
one brings it from one row. With `SqL2`/`L2`/`L1` between codes of the global family nothing reads
the sums, and [`GlobalCodesDatabase`](@ref) is the better layout.

`PackedQuantDatabase(db)` copies `db` into rows (on a `BlockMatrixDatabase` when `storage` is
one, so the result grows); the second constructor takes rows back as they are, `E` being the
shared `SQMinC` or `nothing`.
"""
struct PackedQuantDatabase{B,P<:Union{SQMinC,Nothing},DB<:AbstractDatabase} <: AbstractDatabase
    E::P
    Q::DB
    dim::Int

    function PackedQuantDatabase{B,P,DB}(E::P, Q::DB, dim::Integer) where {B,P<:Union{SQMinC,Nothing},DB<:AbstractDatabase}
        B in (2, 4, 8) || throw(ArgumentError("PackedQuantDatabase: B=$B must be 2, 4 or 8"))
        cpb = codesperbyte(Val(B))
        dim % cpb == 0 ||
            throw(ArgumentError("PackedQuantDatabase: dim=$dim must be a multiple of $cpb ($cpb coordinates are packed per UInt8 at $B bits); pad the vectors"))
        new{B,P,DB}(E, Q, Int(dim))
    end
end

"Bytes of codes, of codes plus padding (where the footer starts, minus one), and of a whole row."
@inline _packedcodebytes(B::Val, dim::Integer) = dim ÷ codesperbyte(B)
@inline _packedoffset(B::Val, dim::Integer) = 4cld(_packedcodebytes(B, dim), 4)
@inline _packedrowbytes(B::Val, dim::Integer, global_::Bool) = _packedoffset(B, dim) + (global_ ? 8 : 16)

function PackedQuantDatabase{B}(E::Union{SQMinC,Nothing}, Q::Union{AbstractMatrix{UInt8},AbstractDatabase}; dim::Integer) where {B}
    Qdb = _asdb(Q)
    rb = _packedrowbytes(Val(B), dim, E isa SQMinC)
    length(Qdb) == 0 || length(Qdb[1]) == rb ||
        throw(ArgumentError("PackedQuantDatabase: rows of $(length(Qdb[1])) bytes, $rb expected for dim=$dim at $B bits"))
    PackedQuantDatabase{B,typeof(E),typeof(Qdb)}(E, Qdb, dim)
end

function PackedQuantDatabase(db::QuantDatabase{B,P}; storage=MatrixDatabase) where {B,P}
    global_ = P <: SQMinC
    n = length(db)
    R = Matrix{UInt8}(undef, _packedrowbytes(Val(B), db.dim, global_), n)
    E = global_ ? db.E : nothing
    _spread(n) do r
        for i in r
            _writerow!(view(R, :, i), db[i], db.dim, global_)
        end
    end
    Q = storage === MatrixDatabase ? MatrixDatabase(R) : append_items!(storage(size(R, 1), UInt8), MatrixDatabase(R))
    PackedQuantDatabase{B,typeof(E),typeof(Q)}(E, Q, db.dim)
end

# a footer value: one load from a contiguous row, four bytes put together from anything else
@inline function _loadf32(row::SIMD.FastContiguousArray{UInt8,1}, o::Int)
    GC.@preserve row unsafe_load(Ptr{Float32}(pointer(row, o)))
end
@inline _loadf32(row::AbstractVector{UInt8}, o::Int) =
    @inbounds reinterpret(Float32, UInt32(row[o]) | UInt32(row[o+1]) << 8 | UInt32(row[o+2]) << 16 | UInt32(row[o+3]) << 24)

@inline function _storef32!(row::AbstractVector{UInt8}, o::Int, x::Float32)
    u = reinterpret(UInt32, x)
    @inbounds for k in 0:3
        row[o+k] = (u >> 8k) % UInt8
    end
end

"Writes `v`'s codes, the padding and its footer into `row`."
function _writerow!(row::AbstractVector{UInt8}, v::SQVec{B}, dim::Int, global_::Bool) where {B}
    nb = _packedcodebytes(Val(B), dim)
    o = _packedoffset(Val(B), dim)
    copyto!(row, 1, v.V, 1, nb)
    @inbounds for k in nb+1:o
        row[k] = 0x00
    end
    _storef32!(row, o + 1, v.Sa)
    _storef32!(row, o + 5, v.Saa)
    if !global_
        _storef32!(row, o + 9, v.E.min)
        _storef32!(row, o + 13, v.E.c)
    end
    row
end

codewidth(::PackedQuantDatabase{B}) where {B} = B
isglobal(::PackedQuantDatabase{B,P}) where {B,P} = P <: SQMinC
quantizer(db::PackedQuantDatabase{B}) where {B} = _quantmodule(B, isglobal(db))

Base.length(db::PackedQuantDatabase) = length(db.Q)
Base.eltype(db::PackedQuantDatabase{B}) where {B} = length(db) > 0 ? typeof(db[1]) : SQVec{B}

Base.@propagate_inbounds function Base.getindex(db::PackedQuantDatabase{B,P}, i::Integer) where {B,P}
    row = db.Q[i]
    o = _packedoffset(Val(B), db.dim)
    E = P <: SQMinC ? db.E : SQMinC(_loadf32(row, o + 9), _loadf32(row, o + 13))
    SQVec{B}(E, view(row, 1:_packedcodebytes(Val(B), db.dim)), _loadf32(row, o + 1), _loadf32(row, o + 5))
end

function show(io::IO, db::PackedQuantDatabase{B}; prefix="", indent="  ") where {B}
    println(io, prefix, isglobal(db) ? "PackedQuantDatabase{$B} (global):" : "PackedQuantDatabase{$B} (per-vector):")
    prefix = prefix * indent
    println(io, prefix, "dim: ", db.dim)
    println(io, prefix, "length: ", length(db))
    println(io, prefix, "storage: ", typeof(db.Q))
end

quantize(db::PackedQuantDatabase{B,Nothing}, v::AbstractVector) where {B} = (_checkdim(db, v); SQVec{B}(v))
function quantize(db::PackedQuantDatabase{B,SQMinC}, v::AbstractVector) where {B}
    _checkdim(db, v)
    codes = Vector{UInt8}(undef, db.dim ÷ codesperbyte(Val(B)))
    packcodes!(Val(B), codes, v, db.E.min, _globalscale(db.E))
    SQVec{B}(db.E, codes)
end

_asstored(db::PackedQuantDatabase, v::AbstractVector) = quantize(db, v)
function _asstored(db::PackedQuantDatabase{B}, v::_SQ{B}) where {B}
    _checkdim(db, v)
    isglobal(db) && v.E != db.E &&
        throw(ArgumentError("push_item!: the vector was quantized with other parameters than the database's ($(v.E) against $(db.E))"))
    _withsums(v)
end

function push_item!(db::PackedQuantDatabase{B}, v) where {B}
    row = Vector{UInt8}(undef, _packedrowbytes(Val(B), db.dim, isglobal(db)))
    push_item!(db.Q, _writerow!(row, _asstored(db, v), db.dim, isglobal(db)))
    db
end

# into block storage the rows are written in place, by all threads (see `set_page_spread!`)
function _appendby!(db::PackedQuantDatabase, m::Integer, item::F) where {F}
    Q = db.Q
    (Q isa BlockMatrixDatabase && _spreads(m)) || return _pushall!(db, m, item)
    n0 = length(Q)
    _reserve!(Q, m)
    Q.len[] = n0 + m
    try
        _spread(m) do r
            for k in r
                _writerow!(Q[n0 + k], _asstored(db, item(k)), db.dim, isglobal(db))
            end
        end
    catch
        Q.len[] = n0
        rethrow()
    end
    db
end

function append_items!(db::PackedQuantDatabase, items)
    items isa Union{AbstractVector,AbstractDatabase} && return _appendby!(db, length(items), k -> items[k])
    for v in items
        push_item!(db, v)
    end

    db
end

append_items!(db::PackedQuantDatabase, X::AbstractMatrix) = append_items!(db, eachcol(X))

function Base.:(==)(a::PackedQuantDatabase{B}, b::PackedQuantDatabase{B}) where {B}
    length(a) == length(b) && a.dim == b.dim && a.E == b.E || return false
    all(i -> a.Q[i] == b.Q[i], eachindex(a))
end

@inline prefetchable(db::PackedQuantDatabase) = prefetchable(db.Q)
@inline prefetch_item(db::PackedQuantDatabase, i::Integer) = prefetch_item(db.Q, i)

function spreadcopy(db::PackedQuantDatabase{B}) where {B}
    Q = spreadcopy(db.Q)
    PackedQuantDatabase{B,typeof(db.E),typeof(Q)}(db.E, Q, db.dim)
end
