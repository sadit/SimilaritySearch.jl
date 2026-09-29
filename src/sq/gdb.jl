# This file is a part of SimilaritySearch.jl

export GlobalQuantDatabase

"""
    GlobalQuantDatabase(bits::Integer, X::AbstractMatrix; minmax=nothing, kwargs...)
    GlobalQuantDatabase(bits::Integer, Q::Matrix{UInt8}, minmax)

An [`AbstractDatabase`](@ref) of vectors quantized to `bits` bits per coordinate (2, 4 or 8)
under **one** `min`/scale pair shared by the whole dataset, keeping that pair -- and the
per-vector code sums -- alongside the codes.

That is the difference from calling [`SQgu8.quantize`](@ref ScalarQuant.SQgu8.quantize)
directly, which hands back a bare `Matrix{UInt8}` and leaves `minmax` to the caller. Without
the parameters a stored matrix of codes cannot be dequantized at all, so it can only ever be
compared against other codes from the same run; with them, a query may stay in its original
`Float32` form.

Indexing yields an [`SQVec`](@ref) of the same width the per-vector quantizers produce -- a
globally quantized vector *is* a per-vector one whose scale happens to be shared -- so every
distance defined for those works here unchanged, and each takes the path it should:

- `SqL2`/`L2` between two stored vectors hit the equal-scale branch, which is an exact
  integer pass over the codes;
- `SqL2`/`L2`/`L1`/`NormCosine` against a plain `Float32` vector take the mixed kernels;
- [`Cosine`](@ref) uses the stored sums for both the dot product's offset terms and the
  norms (issue #77).

# Arguments
- `bits`: 2, 4 or 8
- `X`: the matrix to quantize, one column per vector; `size(X, 1)` must be a multiple of the
  coordinates packed per byte (2 at 4 bits, 4 at 2 bits), or an `ArgumentError` is thrown --
  pad `X`, and then pad every plain vector compared against the database the same way
- `Q`, `minmax`: already-quantized codes and the pair they were produced with, for
  reconstructing a stored database; the sums are recomputed from `Q`

# Keyword Arguments
- `minmax`: the `(min, max)` pair to quantize with; estimated from a sample of `X` by
  [`sqrange`](@ref) when not given, exactly as the underlying `quantize` does
"""
struct GlobalQuantDatabase{BITS} <: AbstractDatabase
    Q::Matrix{UInt8}
    E::SQMinC                 # shared: a code `q` dequantizes to `q * E.c + E.min`
    Sa::Vector{Float32}       # per column: Σ codes, Σ codes² -- derived from Q, so they are
    Saa::Vector{Float32}      # recomputed rather than stored, like the per-column databases
end

function _gqcheckbits(bits)
    bits in (2, 4, 8) || throw(ArgumentError("GlobalQuantDatabase: bits=$bits must be 2, 4 or 8"))
    Val(Int(bits))
end

"The shared dequantization parameters for the range `(min, max)` at width `B`."
function _gqparams(B::Val, minmax)
    mn, mx = Float32(first(minmax)), Float32(last(minmax))
    # `sqglobalscale` is the *quantization* multiplier; a code dequantizes with its inverse
    SQMinC(mn, 1f0 / sqglobalscale(levels(B), mn, mx))
end

function GlobalQuantDatabase(bits::Integer, Q::Matrix{UInt8}, minmax)
    B = _gqcheckbits(bits)
    E = _gqparams(B, minmax)
    n = size(Q, 2)
    Sa = Vector{Float32}(undef, n)
    Saa = Vector{Float32}(undef, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        Sa[i], Saa[i] = codesums(B, view(Q, :, i))
    end

    GlobalQuantDatabase{Int(bits)}(Q, E, Sa, Saa)
end

function GlobalQuantDatabase(bits::Integer, X::AbstractMatrix; minmax=nothing, kwargs...)
    B = _gqcheckbits(bits)
    m, n = size(X)
    cpb = codesperbyte(B)
    m % cpb == 0 ||
        throw(ArgumentError("GlobalQuantDatabase: size(X, 1) = $m must be a multiple of $cpb ($cpb coordinates are packed per UInt8 at $bits bits); pad X"))
    mm = minmax === nothing ? _gqminmax(X, bits; kwargs...) : minmax
    E = _gqparams(B, mm)
    # Every vector, now or later (`quantize(db, v)`), is quantized with the multiplier read
    # back off `E`, so a vector pushed after the fact gets the codes it would have gotten here.
    s = 1f0 / E.c
    Q = Matrix{UInt8}(undef, m ÷ cpb, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        packcodes!(B, view(Q, :, i), view(X, :, i), E.min, s)
    end

    GlobalQuantDatabase(bits, Q, mm)
end

"Estimates the global range the same way the underlying `quantize` does when none is given."
_gqminmax(X::AbstractMatrix, bits::Integer; quant=nothing, samplesize=0) =
    sqrange(vec(X), (1 << bits) - 1; quant, samplesize)

Base.length(db::GlobalQuantDatabase) = size(db.Q, 2)
Base.eltype(db::GlobalQuantDatabase) = typeof(db[1])

Base.@propagate_inbounds Base.getindex(db::GlobalQuantDatabase{BITS}, i::Integer) where {BITS} =
    SQVec{BITS}(db.E, view(db.Q, :, i), db.Sa[i], db.Saa[i])

"""
    quantize(db::GlobalQuantDatabase, v::AbstractVector)

Quantizes `v` with `db`'s own parameters, so the result is comparable with what `db` stores
(unlike the per-column quantizers, where each vector brings its own scale). `length(v)`
must be the database's (padded) dimension.
"""
function quantize(db::GlobalQuantDatabase{BITS}, v::AbstractVector) where BITS
    B = Val(BITS)
    expected = codesperbyte(B) * size(db.Q, 1)
    length(v) == expected ||
        throw(ArgumentError("quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    codes = Vector{UInt8}(undef, size(db.Q, 1))
    packcodes!(B, codes, v, db.E.min, 1f0 / db.E.c)
    SQVec{BITS}(db.E, codes)
end
