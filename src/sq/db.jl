# This file is a part of SimilaritySearch.jl

import ..SimilaritySearch: push_item!, append_items!, show, _appendby!, spreadcopy
using ..SimilaritySearch: MatrixDatabase, BlockMatrixDatabase, MMapMatrixDatabase, _pushall!, _reserve!, _spread, _spreads

export QuantDatabase

"""
    QuantDatabase{B,P,DB<:AbstractDatabase} <: AbstractDatabase

    QuantDatabase{B}(E, Q; dim=nothing, Sa=nothing, Saa=nothing)

An [`AbstractDatabase`](@ref) of vectors quantized to `B` bits per coordinate (2, 4 or 8):
the packed codes live in *any* database of `UInt8` vectors, `Q::DB`, next to what
dequantizes them, `E::P`, which is either one [`SQMinC`](@ref) per vector (`P` a
`Vector{SQMinC}`: the **per-vector** family, [`SQu8Database`](@ref SimilaritySearch.ScalarQuant.SQu8.SQu8Database)
and its siblings) or a single one shared by the whole database (`P == SQMinC`: the
**global** family, [`GlobalQuantDatabase`](@ref)). Indexing yields an [`SQVec`](@ref),
so every distance in this module applies to both families alike. See the module
docstring for how to choose between them, and the aliases above for the constructors
that quantize a matrix.

# Storage

`Q` decides what the database can do, exactly as it does for unquantized vectors:

- [`MatrixDatabase`](@ref), the default of every constructor that quantizes a matrix:
  static and fastest;
- [`BlockMatrixDatabase`](@ref): grows a block at a time. `push_item!`/`append_items!`
  quantize each incoming vector with the database's own parameters and push its codes, so
  a `SearchGraph` can be built over a quantized database one item at a time;
- [`MMapMatrixDatabase`](@ref): grows too, on disk, and outlives the process. Reopen the
  file and hand it back to the constructor with the parameters to get the same database;
- a [`VectorDatabase`](@ref SimilaritySearch.VectorDatabase) of `Vector{UInt8}`, or anything else indexable: whatever a
  custom store needs.

A constructor that quantizes a matrix takes `storage`, a function from the `Matrix{UInt8}`
of codes to the database that will hold them (`MatrixDatabase`, `BlockMatrixDatabase`, or
a closure that fills an mmap file). The constructor above takes codes back as they are --
a matrix, wrapped in a `MatrixDatabase`, or any database -- and an *empty* database of the
right row size is how a growable one starts:

```julia
julia> using SimilaritySearch, SimilaritySearch.ScalarQuant

julia> X = randn(Float32, 64, 1000);

julia> db = GlobalQuantDatabase(8, BlockMatrixDatabase(64, UInt8), extrema(X));   # empty

julia> append_items!(db, MatrixDatabase(X)); length(db)
1000

julia> db == GlobalQuantDatabase(8, X; minmax=extrema(X))      # same codes, byte for byte
```

Growth is serial, like every other database here: concurrent `push_item!`s need a lock.

# Fields

- `E`, `Q`: the two that make a database, and the two that have to be persisted;
- `dim`: the dimension of the vectors, a multiple of the coordinates packed per byte
  (2 at 4 bits, 4 at 2 bits). A dimension that does not fill its last byte is rejected:
  pad the vectors, and then pad every plain vector compared against the database the same
  way, since the mixed distances index it positionally;
- `Sa`, `Saa`: per vector, `Σ codes` and `Σ codes²`, which every distance reads (see the
  note above `codesums` in `codes.jl`). They are sums of the *codes*, measured when a vector
  is quantized: sums of the original `Float32` vector, though more accurate, lost 0.01 to
  0.41 of recall@10 on the SISAP 2025 benchmarks, because quantization rescales each vector
  slightly and only the code sums carry that rescaling (issue #87). Being derived from the
  codes they are recomputed on reconstruction unless passed in, which skips one pass over
  the codes when an mmap-backed database is reopened.

# Arguments
- `E`: a `Vector{SQMinC}` with one entry per stored vector, or one `SQMinC` for all
- `Q`: the codes, as an `AbstractMatrix{UInt8}` (one column per vector) or an
  `AbstractDatabase` of `UInt8` vectors

# Keyword Arguments
- `dim`: the vector dimension; read off `Q` when not given, which an empty
  `VectorDatabase` cannot provide
- `Sa`, `Saa`: the code sums, when the caller kept them; recomputed from `Q` otherwise
"""
struct QuantDatabase{B,P,DB<:AbstractDatabase} <: AbstractDatabase
    E::P
    Q::DB
    dim::Int
    Sa::Vector{Float32}
    Saa::Vector{Float32}

    function QuantDatabase{B,P,DB}(E::P, Q::DB, dim::Integer, Sa::Vector{Float32}, Saa::Vector{Float32}) where {B,P,DB<:AbstractDatabase}
        B in (2, 4, 8) || throw(ArgumentError("QuantDatabase: B=$B must be 2, 4 or 8"))
        P <: SQMinC || P <: AbstractVector{SQMinC} ||
            throw(ArgumentError("QuantDatabase: E must be one SQMinC or a vector of them, got $P"))
        cpb = codesperbyte(Val(B))
        dim % cpb == 0 ||
            throw(ArgumentError("QuantDatabase: dim=$dim must be a multiple of $cpb ($cpb coordinates are packed per UInt8 at $B bits); pad the vectors"))
        n = length(Q)
        P <: SQMinC || length(E) == n ||
            throw(ArgumentError("QuantDatabase: got $(length(E)) quantization parameters for $n stored vectors; there is exactly one `SQMinC` per stored vector"))
        length(Sa) == n && length(Saa) == n ||
            throw(ArgumentError("QuantDatabase: got $(length(Sa)) and $(length(Saa)) code sums for $n stored vectors"))
        new{B,P,DB}(E, Q, Int(dim), Sa, Saa)
    end
end

"Bytes each stored vector occupies in `Q`, read off the storage even when it is empty."
_rowbytes(Q::MatrixDatabase) = size(Q.matrix, 1)
_rowbytes(::BlockMatrixDatabase{Dim}) where {Dim} = Dim
_rowbytes(::MMapMatrixDatabase{Dim}) where {Dim} = Dim
_rowbytes(Q::AbstractDatabase) = length(Q) > 0 ? length(Q[1]) :
    throw(ArgumentError("QuantDatabase: cannot tell the vector dimension from an empty $(typeof(Q)); pass `dim`"))

_asdb(Q::AbstractDatabase) = Q
_asdb(Q::AbstractMatrix{UInt8}) = MatrixDatabase(Q)

function _codesums(B::Val, Q::AbstractDatabase)
    n = length(Q)
    Sa = Vector{Float32}(undef, n)
    Saa = Vector{Float32}(undef, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        Sa[i], Saa[i] = codesums(B, Q[i])
    end

    Sa, Saa
end

function QuantDatabase{B}(E::Union{SQMinC,AbstractVector{SQMinC}}, Q::Union{AbstractMatrix{UInt8},AbstractDatabase};
        dim=nothing, Sa=nothing, Saa=nothing) where {B}
    Qdb = _asdb(Q)
    d = dim === nothing ? codesperbyte(Val(B)) * _rowbytes(Qdb) : Int(dim)
    if Sa === nothing || Saa === nothing
        Sa, Saa = _codesums(Val(B), Qdb)
    end
    QuantDatabase{B,typeof(E),typeof(Qdb)}(E, Qdb, d, convert(Vector{Float32}, Sa), convert(Vector{Float32}, Saa))
end

### the two families' constructors from a matrix

function (::Type{QuantDatabase{B,Vector{SQMinC}}})(X::AbstractMatrix; storage=MatrixDatabase, range::RangePolicy=DEFAULT_RANGE) where {B}
    m, n = size(X)
    cpb = codesperbyte(Val(B))
    m % cpb == 0 ||
        throw(ArgumentError("quantize: size(X, 1) = $m must be a multiple of $cpb ($cpb coordinates are packed per UInt8 at $B bits); pad X"))
    Q = Matrix{UInt8}(undef, m ÷ cpb, n)
    E = Vector{SQMinC}(undef, n)
    Sa = Vector{Float32}(undef, n)
    Saa = Vector{Float32}(undef, n)
    minbatch = getminbatch(n)
    # resolved, not calibrated: a database grows one vector at a time through `push_item!`, which
    # has no matrix to calibrate on, and a build from a matrix must produce the very codes that
    # growing would. Calibration belongs to `SQEncoder`, which keeps its fitted policy.
    policy = resolverange(range, B)
    @BATCHES minbatch for i in 1:n
        E[i] = quantvector!(Val(B), view(Q, :, i), view(X, :, i); range=policy)
        Sa[i], Saa[i] = codesums(Val(B), view(Q, :, i))
    end

    QuantDatabase{B}(E, storage(Q); dim=m, Sa, Saa)
end

# the per-vector rebuild, under its family's own name
(::Type{QuantDatabase{B,Vector{SQMinC}}})(E::AbstractVector{SQMinC}, Q::Union{AbstractMatrix{UInt8},AbstractDatabase}; kwargs...) where {B} =
    QuantDatabase{B}(convert(Vector{SQMinC}, E), Q; kwargs...)

"The shared dequantization parameters for the range `(min, max)` at width `B`."
function _globalparams(B::Val, minmax)
    mn, mx = Float32(first(minmax)), Float32(last(minmax))
    # `sqglobalscale` is the *quantization* multiplier; a code dequantizes with its inverse
    SQMinC(mn, 1f0 / sqglobalscale(levels(B), mn, mx))
end

"The multiplier every vector of a global database is quantized with: read back off the stored step, so a vector pushed later gets the codes the batch would have given it."
_globalscale(E::SQMinC) = 1f0 / E.c

function (::Type{QuantDatabase{B,SQMinC}})(X::AbstractMatrix; minmax=nothing, storage=MatrixDatabase, quant=nothing, samplesize=0) where {B}
    m, n = size(X)
    cpb = codesperbyte(Val(B))
    m % cpb == 0 ||
        throw(ArgumentError("GlobalQuantDatabase: size(X, 1) = $m must be a multiple of $cpb ($cpb coordinates are packed per UInt8 at $B bits); pad X"))
    mm = minmax === nothing ? sqrange(vec(X), levels(Val(B)); quant, samplesize) : minmax
    E = _globalparams(Val(B), mm)
    s = _globalscale(E)
    Q = Matrix{UInt8}(undef, m ÷ cpb, n)
    Sa = Vector{Float32}(undef, n)
    Saa = Vector{Float32}(undef, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        packcodes!(Val(B), view(Q, :, i), view(X, :, i), E.min, s)
        Sa[i], Saa[i] = codesums(Val(B), view(Q, :, i))
    end

    QuantDatabase{B}(E, storage(Q); dim=m, Sa, Saa)
end

# the global rebuild, under its family's own name: codes plus the range they were made with
(::Type{QuantDatabase{B,SQMinC}})(Q::Union{AbstractMatrix{UInt8},AbstractDatabase}, minmax; kwargs...) where {B} =
    QuantDatabase{B}(_globalparams(Val(B), minmax), Q; kwargs...)

### the database interface

"The code width of the stored vectors, in bits."
codewidth(::QuantDatabase{B}) where {B} = B

"Whether every stored vector shares one `min`/scale pair (the global family)."
isglobal(::QuantDatabase{B,P}) where {B,P} = P <: SQMinC

@inline _param(E::SQMinC, i) = E
Base.@propagate_inbounds _param(E::AbstractVector{SQMinC}, i) = E[i]

Base.length(db::QuantDatabase) = length(db.Q)
Base.eltype(db::QuantDatabase{B}) where {B} = length(db) > 0 ? typeof(db[1]) : SQVec{B}

Base.@propagate_inbounds Base.getindex(db::QuantDatabase{B}, i::Integer) where {B} =
    SQVec{B}(_param(db.E, i), db.Q[i], db.Sa[i], db.Saa[i])

function show(io::IO, db::QuantDatabase{B}; prefix="", indent="  ") where {B}
    println(io, prefix, isglobal(db) ? "GlobalQuantDatabase{$B}:" : "QuantDatabase{$B} (per-vector):")
    prefix = prefix * indent
    println(io, prefix, "dim: ", db.dim)
    println(io, prefix, "length: ", length(db))
    println(io, prefix, "storage: ", typeof(db.Q))
end

function _checkdim(db::QuantDatabase, v)
    length(v) == db.dim ||
        throw(ArgumentError("QuantDatabase: length(v) = $(length(v)) must equal the database's vector dimension ($(db.dim))"))
end

"""
    quantize(db::QuantDatabase, v::AbstractVector) -> SQVec

Quantizes `v` the way `db`'s vectors are: with `db`'s shared parameters in the global family,
so the result is comparable with what it stores; on `v`'s own range (the default policy) in the per-vector one,
where `db` only fixes the dimension. This is what [`push_item!`](@ref) stores, and what a
query must go through to be compared as codes against codes (a `Float32` query needs no
quantization: the mixed distances take it as it is).
"""
function quantize(db::QuantDatabase{B,<:AbstractVector{SQMinC}}, v::AbstractVector) where {B}
    _checkdim(db, v)
    SQVec{B}(v)
end

function quantize(db::QuantDatabase{B,SQMinC}, v::AbstractVector) where {B}
    _checkdim(db, v)
    codes = Vector{UInt8}(undef, db.dim ÷ codesperbyte(Val(B)))
    packcodes!(Val(B), codes, v, db.E.min, _globalscale(db.E))
    SQVec{B}(db.E, codes)
end

function _pushcodes!(db::QuantDatabase, E::SQMinC, codes, Sa, Saa)
    push_item!(db.Q, codes)
    db.E isa AbstractVector && push!(db.E, E)
    push!(db.Sa, Sa)
    push!(db.Saa, Saa)
    db
end

"""
    push_item!(db::QuantDatabase, v::AbstractVector)
    push_item!(db::QuantDatabase, v::SQVec)

Appends `v` to `db`: a plain vector is quantized with [`quantize`](@ref)`(db, v)` first, an
already quantized one is stored as it is (its parameters must be `db`'s in the global
family). The codes go to `db.Q`, which must support growth (`BlockMatrixDatabase`,
`MMapMatrixDatabase`, `VectorDatabase`, ...); a `MatrixDatabase` throws, as it does for
plain vectors.
"""
push_item!(db::QuantDatabase, v::AbstractVector) = push_item!(db, quantize(db, v))

function push_item!(db::QuantDatabase{B}, v::SQVec{B}) where {B}
    v = _asstored(db, v)
    _pushcodes!(db, v.E, v.V, v.Sa, v.Saa)
end

# what `db` stores for an item: a plain vector is quantized, a quantized one is checked
_asstored(db::QuantDatabase, v::AbstractVector) = quantize(db, v)

function _asstored(db::QuantDatabase{B}, v::SQVec{B}) where {B}
    _checkdim(db, v)
    isglobal(db) && v.E != db.E &&
        throw(ArgumentError("push_item!: the vector was quantized with other parameters than the database's ($(v.E) against $(db.E))"))
    v
end

# Bulk append into block storage: the codes, the sums and the per-vector parameters get their room
# first and are written by all threads (see `set_page_spread!`); `item(k)` is quantized or encoded
# inside the parallel loop. Any other storage, or few items, goes through `push_item!`.
function _appendby!(db::QuantDatabase, m::Integer, item::F) where {F}
    Q = db.Q
    (Q isa BlockMatrixDatabase && _spreads(m)) || return _pushall!(db, m, item)
    n0 = length(Q)
    _reserve!(Q, m)
    Q.len[] = n0 + m
    resize!(db.Sa, n0 + m)
    resize!(db.Saa, n0 + m)
    E = db.E
    E isa AbstractVector && resize!(E, n0 + m)
    try
        _spread(m) do r
            for k in r
                v = _asstored(db, item(k))
                i = n0 + k
                Q[i] = v.V
                E isa AbstractVector && (@inbounds E[i] = v.E)
                @inbounds db.Sa[i] = v.Sa
                @inbounds db.Saa[i] = v.Saa
            end
        end
    catch
        Q.len[] = n0
        resize!(db.Sa, n0)
        resize!(db.Saa, n0)
        E isa AbstractVector && resize!(E, n0)
        rethrow()
    end
    db
end

"""
    append_items!(db::QuantDatabase, items)

Appends every object of `items` (an `AbstractDatabase`, an iterator of vectors, or a matrix
whose columns are the vectors) to `db`, as [`push_item!`](@ref) would. Many indexable items into
block storage are quantized and written by all threads (see [`set_page_spread!`](@ref)).
"""
function append_items!(db::QuantDatabase, items)
    items isa Union{AbstractVector,AbstractDatabase} && return _appendby!(db, length(items), k -> items[k])
    for v in items
        push_item!(db, v)
    end

    db
end

append_items!(db::QuantDatabase, X::AbstractMatrix) = append_items!(db, eachcol(X))

"""
    ==(a::QuantDatabase, b::QuantDatabase)

Two quantized databases are equal when they hold the same codes under the same parameters,
whatever storage each one uses.
"""
function Base.:(==)(a::QuantDatabase{B}, b::QuantDatabase{B}) where {B}
    length(a) == length(b) && a.dim == b.dim && a.E == b.E || return false
    for i in eachindex(a)
        a.Q[i] == b.Q[i] || return false
    end

    true
end

Base.:(==)(::QuantDatabase, ::QuantDatabase) = false


# the mixed and symmetric distances read the item's codes and its stored sums (and its own
# quantizer in the per-vector family): all of it is prefetched together
@inline prefetchable(db::QuantDatabase) = prefetchable(db.Q)
@inline function prefetch_item(db::QuantDatabase, i::Integer)
    prefetch_item(db.Q, i)
    _prefetch(pointer(db.Sa, i))
    _prefetch(pointer(db.Saa, i))
    db.E isa AbstractVector && _prefetch(pointer(db.E, i))
    nothing
end

# the codes, the sums and the per-vector parameters, each written by all threads
function spreadcopy(db::QuantDatabase{B}) where {B}
    E = db.E isa AbstractVector ? spreadcopy(db.E) : db.E
    Q = spreadcopy(db.Q)
    QuantDatabase{B,typeof(E),typeof(Q)}(E, Q, db.dim, spreadcopy(db.Sa), spreadcopy(db.Saa))
end
