# This file is a part of SimilaritySearch.jl
#
# A layout of a quantized database besides `QuantDatabase`'s codes-plus-two-arrays: the codes alone.
# Both hold the same codes and give the same distances; they differ in what an evaluation brings from
# memory.
#
# A third layout was measured and not adopted (branch sq-layouts at c3a7aafb, PackedQuantDatabase):
# the sums at the end of each code row, one region per item instead of three. Same graph and visits,
# 64 threads, 8-bit codes of 384 dimensions, QPS at recall 0.9 against `QuantDatabase`: rows of 392
# bytes back to back lost 2.5-4% under SqL2 and 3-12% under NormCosine on ccnews and gooaq, since the
# codes of most rows then start off a 32-byte boundary and the vector loads straddle cache lines;
# padded to 416 or 448 bytes it went from -0.4% to +5% on ccnews (603K) and lost 6-10% on gooaq (3M).
# The separate arrays of sums are 8 bytes per vector, 16 vectors to a cache line, and stay in the
# last-level cache at these sizes, while a longer row costs a line from memory per item.

export GlobalCodesDatabase

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
rotation, 64 threads, tuned for recall 0.9), the same graph searched through both layouts, with
the same visits and answers: 10.6-13.8% more queries per second on ccnews (603K vectors, three
graphs; ~315 -> ~283 ns·thread per evaluation) and 0-6.4% on gooaq (3M, three graphs). Prefetching
the codes alone while still loading the sums lost 3-10%: the loads are what costs. It also saves
8 bytes per vector.

Everything else still works, at a price: `NormCosine`, `Cosine`, codes of other scales and a
prepared [`SQQuery`](@ref) need the sums, and get them from one more pass over the codes per
evaluation (22-28% fewer queries per second under `NormCosine`, measured as above). Use a
`GlobalQuantDatabase` for those, and for an `AsymmetricSearchGraph` whose queries are `SQQuery`s.

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
