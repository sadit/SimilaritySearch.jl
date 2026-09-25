# This file is a part of SimilaritySearch.jl

export GlobalQuantDatabase, Cosine

"""
    Cosine()

Cosine dissimilarity (`1 - cos`) between two quantized vectors, computed from their codes:
the dot product of the dequantized vectors divided by their dequantized norms.

Both halves of that need per-vector quantities the codes alone do not carry, and both are
already stored by every quantized vector in this module (`Sa = Σ codes`, `Saa = Σ codes²`):

- the **dot product** expands to `cA·cB·Σab + cA·mB·Σa + cB·mA·Σb + n·mA·mB`, so dropping
  everything but `Σab` -- as a raw-code dot product does -- only preserves order when the
  offsets are zero. On centered data (any ordinary embedding) it does not: measured over
  20k unit-norm vectors in dim 128, ranking by the raw code dot product gave recall@10 of
  **0.005** against exact cosine, while the full expansion gave **0.97** (issue #77);
- the **norms**, `‖â‖² = c²·Saa + 2·c·m·Sa + n·m²`, correct the drift quantization leaves in
  a vector that was normalized before being quantized. That matters more the fewer bits
  there are: on non-negative data, recall@10 went 0.9685 -> 0.979 at 8 bits, 0.590 -> 0.666
  at 4 bits, and 0.111 -> 0.126 at 2 bits.
"""
struct Cosine <: SemiMetric end

# per-width bridges: the integer kernels live in the per-column modules, which already
# vectorize them (one 32-lane pass, one 16-lane pass, scalar remainder)
@inline _dotcodes(a::SQu8.SQu8Vec, b::SQu8.SQu8Vec) = SQu8.u8dotcodes(a.V, b.V)
@inline _dotcodes(a::SQu4.SQu4Vec, b::SQu4.SQu4Vec) = SQu4.u4dotcodes(a.V, b.V)
@inline _dotcodes(a::SQu2.SQu2Vec, b::SQu2.SQu2Vec) = SQu2.u2dotcodes(a.V, b.V)

"Number of coordinates a quantized vector stands for, padding included."
@inline ncoords(a::SQu8.SQu8Vec) = length(a.V)
@inline ncoords(a::SQu4.SQu4Vec) = 2 * length(a.V)
@inline ncoords(a::SQu2.SQu2Vec) = 4 * length(a.V)

"Dot product of the two dequantized vectors, from their codes -- see [`Cosine`](@ref)."
@inline function quantdot(a, b)::Float64
    cA, mA = Float64(a.E.c), Float64(a.E.min)
    cB, mB = Float64(b.E.c), Float64(b.E.min)
    cA * cB * Float64(_dotcodes(a, b)) + cA * mB * Float64(a.Sa) + cB * mA * Float64(b.Sa) +
        ncoords(a) * mA * mB
end

"Euclidean norm of the dequantized vector, from its stored code sums."
@inline function quantnorm(a)::Float64
    c, m = Float64(a.E.c), Float64(a.E.min)
    sqrt(max(0.0, c * c * Float64(a.Saa) + 2 * c * m * Float64(a.Sa) + ncoords(a) * m * m))
end

function evaluate(::Cosine, a, b)::Float32
    na, nb = quantnorm(a), quantnorm(b)
    (na == 0 || nb == 0) && return 1f0
    Float32(1.0 - clamp(quantdot(a, b) / (na * nb), -1.0, 1.0))
end

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

Indexing yields the same `SQu2Vec`/`SQu4Vec`/`SQu8Vec` the per-column quantizers produce --
a globally quantized vector *is* a per-column one whose scale happens to be shared -- so
every distance defined for those works here unchanged, and each takes the path it should:

- `SqL2`/`L2` between two stored vectors hit the equal-scale branch, which is an exact
  integer pass over the codes;
- `SqL2`/`L2`/`L1` against a plain `Float32` vector take the mixed kernels;
- [`Cosine`](@ref) uses the stored sums for both the dot product's offset terms and the
  norms (issue #77).

# Arguments
- `bits`: 2, 4 or 8
- `X`: the matrix to quantize, one column per vector
- `Q`, `minmax`: already-quantized codes and the pair they were produced with, for
  reconstructing a stored database; the sums are recomputed from `Q`

# Keyword Arguments
- `minmax`: the `(min, max)` pair to quantize with; estimated from quantiles of a sample of
  `X` when not given, exactly as the underlying `quantize` does
"""
struct GlobalQuantDatabase{BITS} <: AbstractDatabase
    Q::Matrix{UInt8}
    E::SQMinC                 # shared: a code `q` dequantizes to `q * E.c + E.min`
    Sa::Vector{Float32}       # per column: Σ codes, Σ codes² -- derived from Q, so they are
    Saa::Vector{Float32}      # recomputed rather than stored, like the per-column databases
end

_gqmod(::Val{2}) = SQu2
_gqmod(::Val{4}) = SQu4
_gqmod(::Val{8}) = SQu8
_gqsums(::Val{8}, v) = SQu8.u8sums(v)
_gqsums(::Val{4}, v) = SQu4.u4sums(v)
_gqsums(::Val{2}, v) = SQu2.u2sums(v)
_gqquantize(::Val{8}, X; kwargs...) = SQgu8.quantize(X; kwargs...)
_gqquantize(::Val{4}, X; kwargs...) = SQgu4.quantize(X; kwargs...)
_gqquantize(::Val{2}, X; kwargs...) = SQgu2.quantize(X; kwargs...)
_gqlevels(::Val{B}) where B = 2^B - 1

function GlobalQuantDatabase(bits::Integer, Q::Matrix{UInt8}, minmax)
    bits in (2, 4, 8) || throw(ArgumentError("GlobalQuantDatabase: bits=$bits must be 2, 4 or 8"))
    B = Val(Int(bits))
    mn, mx = Float32(first(minmax)), Float32(last(minmax))
    # `sqglobalscale` is the *quantization* multiplier; a code dequantizes with its inverse
    E = SQMinC(mn, 1f0 / sqglobalscale(_gqlevels(B), mn, mx))
    n = size(Q, 2)
    Sa = Vector{Float32}(undef, n)
    Saa = Vector{Float32}(undef, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        Sa[i], Saa[i] = _gqsums(B, view(Q, :, i))
    end

    GlobalQuantDatabase{Int(bits)}(Q, E, Sa, Saa)
end

function GlobalQuantDatabase(bits::Integer, X::AbstractMatrix; minmax=nothing, kwargs...)
    bits in (2, 4, 8) || throw(ArgumentError("GlobalQuantDatabase: bits=$bits must be 2, 4 or 8"))
    B = Val(Int(bits))
    mm = minmax === nothing ? _gqminmax(X, bits; kwargs...) : minmax
    Q = _gqquantize(B, X; minmax=mm)
    GlobalQuantDatabase(bits, Q, mm)
end

"Estimates the global range the same way the underlying `quantize` does when none is given."
_gqminmax(X::AbstractMatrix, bits::Integer; quant=nothing, samplesize=0) =
    sqrange(vec(X), (1 << bits) - 1; quant, samplesize)

Base.length(db::GlobalQuantDatabase) = size(db.Q, 2)
Base.eltype(db::GlobalQuantDatabase) = typeof(db[1])

Base.@propagate_inbounds Base.getindex(db::GlobalQuantDatabase{8}, i::Integer) =
    SQu8.SQu8Vec(db.E, view(db.Q, :, i), db.Sa[i], db.Saa[i])
Base.@propagate_inbounds Base.getindex(db::GlobalQuantDatabase{4}, i::Integer) =
    SQu4.SQu4Vec(db.E, view(db.Q, :, i), db.Sa[i], db.Saa[i])
Base.@propagate_inbounds Base.getindex(db::GlobalQuantDatabase{2}, i::Integer) =
    SQu2.SQu2Vec(db.E, view(db.Q, :, i), db.Sa[i], db.Saa[i])

"""
    quantize(db::GlobalQuantDatabase, v::AbstractVector)

Quantizes `v` with `db`'s own parameters, so the result is comparable with what `db` stores
(unlike the per-column quantizers, where each vector brings its own scale).
"""
function quantize(db::GlobalQuantDatabase{BITS}, v::AbstractVector) where BITS
    B = Val(BITS)
    Q = _gqquantize(B, reshape(collect(Float32, v), :, 1); minmax=(db.E.min, db.E.min + db.E.c * _gqlevels(B)))
    codes = view(Q, :, 1)
    Sa, Saa = _gqsums(B, codes)
    BITS == 8 ? SQu8.SQu8Vec(db.E, codes, Sa, Saa) :
    BITS == 4 ? SQu4.SQu4Vec(db.E, codes, Sa, Saa) :
                SQu2.SQu2Vec(db.E, codes, Sa, Saa)
end
