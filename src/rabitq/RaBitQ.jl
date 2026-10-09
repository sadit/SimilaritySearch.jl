# This file is a part of SimilaritySearch.jl

"""
    RaBitQ

The RaBitQ estimator (Gao & Long, 2024) as an [`AbstractEstimator`](@ref) for an
[`AsymmetricSearchGraph`](@ref):
every object is stored as the sign bits of its randomly rotated vector plus three scalars,
and a raw query is evaluated against those bits with an unbiased estimate of the inner
product whose error bound is known per object.

    est = RaBitQCosine(Projections.qr(dim, dim))    # or RaBitQL2; RandomizedHadamard(dim) rotates in dim log dim
    G = AsymmetricSearchGraph(est, rabitqcodes(est))  # a growable storage of codes
    append_items!(G, ctx, MatrixDatabase(X))        # raw vectors in, codes stored
    search(G, ctx, q, knnqueue(KnnSorted, 10))      # raw query in, rotated once inside

The graph rotates each query once (`encodequery`), encodes each inserted item once
(`encode`), and evaluates the estimator between the rotated query and the stored bits; it
also evaluates two stored codes against each other, which its neighborhood filters need,
with the SimHash estimate over the Hamming distance of the bits.
"""
module RaBitQ

using LinearAlgebra, Random, SIMD
using Statistics: quantile
using ..SimilaritySearch
using ..SimilaritySearch: AbstractEstimator, @BATCHES, PreMetric, AbstractDatabase, rotate, rotationdim, rotationname
using ..SimilaritySearch.Projections: RandomProjections, RandomizedHadamard, Rotation
using ..SimilaritySearch.ScalarQuant: SQVec, SQMinC, dotmixed, quantnorm, _quantparams, _rotatedsample, _quantcode, _quantmodule, RangePolicy, AutoRange
using ..SimilaritySearch.Dist.Bits: Hamming
import ..SimilaritySearch: encode, encodequery, evaluate
import ..SimilaritySearch.ScalarQuant: quantizer, codewidth, isglobal

export RaBitQCosine, RaBitQL2, RaBitQCode, RaBitQQuery, rabitqcodes, estimatecos, errorbound

"""
    RaBitQCode{V<:AbstractVector{UInt64}}

What is stored per object: the sign bits of its rotated vector (`dim` bits in `cld(dim, 64)`
words, low bit first) and three scalars the estimator needs -- `c = ⟨ō, o⟩ / ‖o‖`, the
projection of the unit vector onto its own sign vector, which normalizes the estimate;
`norm = ‖o‖`, for the Euclidean distance; and `err`, the scale of the estimate's confidence
interval, `sqrt((1 - c²) / c²) · 1.9 / sqrt(dim - 1)`.
"""
struct RaBitQCode{V<:AbstractVector{UInt64}}
    bits::V
    c::Float32
    norm::Float32
    err::Float32
end

"""
    RaBitQQuery{V<:AbstractVector{Float32}}

A raw query prepared once for the estimator: rotated and normalized to a unit vector (`r`),
with the raw norm (`norm`) kept for the Euclidean distance.
"""
struct RaBitQQuery{V<:AbstractVector{Float32}}
    r::V
    norm::Float32
end

"""
    RaBitQCosine(rotation)
    RaBitQL2(rotation)

The RaBitQ estimators of the cosine dissimilarity (`1 - cos`) and of the Euclidean distance
between a raw query and a stored code, over `rotation`, a [`SimilaritySearch.Projections.Rotation`](@ref):
`Projections.qr(dim, dim)` or `RandomizedHadamard(dim)`. The dimension is read off it. The
rotation is not optional here: the estimate is unbiased and its error bound holds because the
sign vector is taken in a uniformly random basis.

Both are `AbstractEstimator`s: `encode(est, o)` gives the [`RaBitQCode`](@ref) the storage
receives, `encodequery(est, q)` the [`RaBitQQuery`](@ref) a raw query becomes, and
`evaluate(est, q::RaBitQQuery, o::RaBitQCode)` the estimated distance. Between two codes,
`evaluate(est, a::RaBitQCode, b::RaBitQCode)` estimates the cosine from the Hamming distance
of the bits, `cos(π h / dim)`, which is what the graph's neighborhood filters use.
"""
abstract type AbstractRaBitQ <: AbstractEstimator end

struct RaBitQCosine{ROT} <: AbstractRaBitQ
    rot::ROT
    dim::Int
    m::Float32      # 1 / sqrt(dim): the entries of the sign vector ō, so that ‖ō‖ = 1
end

struct RaBitQL2{ROT} <: AbstractRaBitQ
    rot::ROT
    dim::Int
    m::Float32
end

RaBitQCosine(rot::Rotation) = (d = rotationdim(rot); RaBitQCosine(rot, d, Float32(1 / sqrt(d))))
RaBitQL2(rot::Rotation) = (d = rotationdim(rot); RaBitQL2(rot, d, Float32(1 / sqrt(d))))

"""
    rabitqcodes(est) -> VectorDatabase

An empty, growable storage of the codes `est` produces, for an `AsymmetricSearchGraph` over
it: `RaBitQCode`s for the bit estimators, `(RaBitQCode, fine)` tuples for a
[`RaBitQRefined`](@ref).
"""
rabitqcodes(::AbstractRaBitQ) = VectorDatabase(type=RaBitQCode{Vector{UInt64}})
rabitqcodes() = rabitqcodes(RaBitQCosine)
rabitqcodes(::Type{<:AbstractRaBitQ}) = VectorDatabase(type=RaBitQCode{Vector{UInt64}})

### encoding

@inline _setbit!(bits::AbstractVector{UInt64}, i::Int) = @inbounds bits[((i - 1) >>> 6) + 1] |= one(UInt64) << ((i - 1) & 63)

_checkdim(e::AbstractRaBitQ, v) = length(v) == e.dim ||
    throw(ArgumentError("RaBitQ: a vector of dimension $(length(v)) for an estimator of dimension $(e.dim)"))

"""
    encode(e, o) -> RaBitQCode

The sign bits of the rotated `o`, `c = ⟨ō, o⟩ / ‖o‖` (scale-free, so it does not depend on
how the rotation scales its output), `‖o‖` of the raw vector, and the error scale.
"""
function encode(e::AbstractRaBitQ, o::AbstractVector)
    _checkdim(e, o)
    _encode_rotated(e, rotate(e.rot, o), Float32(sqrt(sum(abs2, o))))
end

"The code of an already rotated vector `r` whose raw norm is `onorm`."
function _encode_rotated(e::AbstractRaBitQ, r::AbstractVector{Float32}, onorm::Float32)
    D = e.dim
    bits = zeros(UInt64, cld(D, 64))
    s1 = 0.0f0; s2 = 0.0f0
    @inbounds for i in 1:D
        x = r[i]
        x >= 0 && _setbit!(bits, i)
        s1 += abs(x)
        s2 += x * x
    end
    nr = sqrt(s2)
    c = nr > 0 ? min(1f0, e.m * s1 / nr) : 1f0            # ⟨ō, o⟩ / ‖o‖, with ō = sign(r) / sqrt(D)
    err = c > 0 ? Float32(sqrt((1 - c * c) / (c * c)) * 1.9 / sqrt(D - 1)) : Inf32
    RaBitQCode(bits, c, onorm, err)
end

"""
    encodequery(e, q) -> RaBitQQuery

The rotated `q`, normalized to a unit vector so that the estimate below is one signed sum
per pair, and `‖q‖` of the raw vector for the Euclidean distance.
"""
function encodequery(e::AbstractRaBitQ, q::AbstractVector)
    _checkdim(e, q)
    r = rotate(e.rot, q)
    nr = sqrt(sum(abs2, r))
    nr > 0 && (r ./= nr)
    RaBitQQuery(r, Float32(sqrt(sum(abs2, q))))
end

"""
    rabitqcodes(e::AbstractRaBitQ, X::AbstractMatrix; minbatch=4) -> VectorDatabase

Encodes every column of `X` in parallel into a growable storage of codes.
"""
function rabitqcodes(e::AbstractRaBitQ, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)
    codes = Vector{RaBitQCode{Vector{UInt64}}}(undef, n)
    @BATCHES minbatch for i in 1:n
        codes[i] = encode(e, view(X, :, i))
    end

    VectorDatabase(codes)
end

### the estimate

### The one pass the estimate costs: `m · Σ ±r_i`, the sign of each term read from the bits.
###
### Each 64-bit word is expanded into four 16-lane masks by shifting the broadcast word by
### the lane index and testing the low bit, and the mask selects `x` or `-x` for the lanes.
### Measured against a scalar loop that reads one bit per iteration (32768 codes, one thread):
### 66.9 ns against 268.6 at 384 dimensions, 83.1 against 360.0 at 512, 144.0 against 725.4
### at 1024. A byte-indexed table of 8 signs with one fma per 8 floats (8 KB, kept in L1)
### tied with it at 384 and 512 (64.2 and 82.8) and won 8% at 1024 with two accumulators;
### the mask needs no table. Both differ from the scalar sum by 1e-5, the order of the
### additions.

const _LANES = Vec{16,UInt64}(ntuple(i -> UInt64(i - 1), 16))

@inline function _signeddot(bits::AbstractVector{UInt64}, r::AbstractVector{Float32}, m::Float32)::Float32
    n = length(r)
    acc = zero(Vec{16,Float32})
    i = 1; wi = 1
    @inbounds while i + 63 <= n
        w = Vec{16,UInt64}(bits[wi])
        for c in 0:3
            mask = ((w >>> (_LANES + UInt64(16c))) & one(UInt64)) == one(UInt64)
            x = vload(Vec{16,Float32}, r, i + 16c)
            acc += vifelse(mask, x, -x)
        end
        i += 64; wi += 1
    end
    d = sum(acc)
    @inbounds while i <= n            # the last, partial word
        w = bits[((i - 1) >>> 6) + 1]
        b = (w >>> ((i - 1) & 63)) & one(UInt64)
        x = r[i]
        d += ifelse(b === one(UInt64), x, -x)
        i += 1
    end

    d * m
end

"""
    estimatecos(e::AbstractRaBitQ, q::RaBitQQuery, o::RaBitQCode) -> Float32

The RaBitQ estimate of the cosine between the raw query behind `q` and the object behind `o`:
`⟨ō, q⟩ / (⟨ō, o⟩ ‖q‖)`, unbiased, with a confidence interval of half-width about
[`errorbound`](@ref)`(e, o)` (for unit vectors).
"""
@inline estimatecos(e::AbstractRaBitQ, q::RaBitQQuery, o::RaBitQCode)::Float32 =
    _signeddot(o.bits, q.r, e.m) / o.c

"""
    errorbound(e, o::RaBitQCode) -> Float32

The half-width of the estimate's confidence interval for the object behind `o`,
`sqrt((1 - c²) / c²) · ε₀ / sqrt(dim - 1)` with `ε₀ = 1.9`: the true cosine lies within it
of the estimate with probability about `1 - exp(-ε₀² / 2) ≈ 0.84` (measured at 0.80 on unit
Gaussian vectors in 384-d). Larger for objects their sign vector represents worse.
"""
@inline errorbound(::AbstractRaBitQ, o::RaBitQCode) = o.err

"The SimHash estimate of the cosine between two stored objects, from the Hamming distance of their bits."
@inline function _codecos(e::AbstractRaBitQ, a::RaBitQCode, b::RaBitQCode)::Float32
    h = evaluate(Hamming(), a.bits, b.bits)
    cos(Float32(pi) * h / e.dim)
end

@inline evaluate(e::RaBitQCosine, q::RaBitQQuery, o::RaBitQCode)::Float32 = 1f0 - estimatecos(e, q, o)
@inline evaluate(e::RaBitQCosine, a::RaBitQCode, b::RaBitQCode)::Float32 = 1f0 - _codecos(e, a, b)

@inline function _l2(na::Float32, nb::Float32, c::Float32)::Float32
    sqrt(max(0f0, na * na + nb * nb - 2f0 * na * nb * c))
end

@inline evaluate(e::RaBitQL2, q::RaBitQQuery, o::RaBitQCode)::Float32 = _l2(o.norm, q.norm, estimatecos(e, q, o))
@inline evaluate(e::RaBitQL2, a::RaBitQCode, b::RaBitQCode)::Float32 = _l2(a.norm, b.norm, _codecos(e, a, b))

# the order every index in SimilaritySearch uses is query first; the other one is accepted too
@inline evaluate(e::AbstractRaBitQ, o::RaBitQCode, q::RaBitQQuery) = evaluate(e, q, o)

function Base.show(io::IO, e::AbstractRaBitQ)
    print(io, nameof(typeof(e)), "(dim=", e.dim, ", rotation=", rotationname(e.rot), ")")
end

include("refined.jl")

end
