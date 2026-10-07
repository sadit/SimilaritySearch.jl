# This file is a part of SimilaritySearch.jl
#
# SQEncoder: the scalar quantizers as the encoder of an AsymmetricSearchGraph, with an
# optional rotation in front -- and what any other level built on these quantizers shares
# with it (RaBitQ's RaBitQVectorFallback): the quantizer is named by its module, which says
# the family and the width at once, and the rotation by the object that applies it.

export SQEncoder, sqcodes, quantizer

using Random
using ..SimilaritySearch: AbstractEstimator, rotate, rotationdim, rotationname, BlockMatrixDatabase
import ..SimilaritySearch: encode, encodequery

"""
    quantizer(x) -> Module

The quantizer module `x` was built with, one of `SQgu8`, `SQgu4`, `SQgu2`, `SQu8`, `SQu4`,
`SQu2`: the family and the width in one name. Defined for a [`QuantDatabase`](@ref), a
[`SQEncoder`](@ref) and `RaBitQ.RaBitQVectorFallback`.
"""
quantizer(::QuantDatabase{B,P}) where {B,P} = _quantmodule(B, P <: SQMinC)

"The `(bits, global?)` a ScalarQuant quantizer module stands for."
function _quantspec(quant::Module)
    quant === SQgu8 && return (8, true)
    quant === SQgu4 && return (4, true)
    quant === SQgu2 && return (2, true)
    quant === SQu8 && return (8, false)
    quant === SQu4 && return (4, false)
    quant === SQu2 && return (2, false)
    throw(ArgumentError("quant=$quant must be one of ScalarQuant's SQgu8, SQgu4, SQgu2, SQu8, SQu4, SQu2"))
end

"The module back from what a type carries."
_quantmodule(B::Int, global_::Bool) =
    global_ ? (B == 8 ? SQgu8 : B == 4 ? SQgu4 : SQgu2) : (B == 8 ? SQu8 : B == 4 ? SQu4 : SQu2)

_globalparams(B::Int, mn::Float32, mx::Float32) = SQMinC(mn, 1f0 / sqglobalscale(levels(Val(B)), mn, mx))

"""
    _quantparams(quant, sample) -> (B, E)
    _quantparams(quant, minmax) -> (B, E)

The width of `quant` and its parameters: the global `SQMinC`, fitted by `sqautorange` on
`sample()` (only called for a global module) or given as `minmax`, or `nothing` for a
per-vector module, whose ranges come with each code.
"""
function _quantparams(quant::Module, sample::Function)
    B, global_ = _quantspec(quant)
    global_ || return B, nothing
    mn, mx = sqautorange(sample(), levels(Val(B)))
    B, _globalparams(B, mn, mx)
end

function _quantparams(quant::Module, minmax)
    B, global_ = _quantspec(quant)
    global_ || return B, nothing
    minmax === nothing && throw(ArgumentError("$(nameof(quant)) needs `minmax`, or a matrix to estimate its range from"))
    B, _globalparams(B, Float32(minmax[1]), Float32(minmax[2]))
end

"A sample of `X`'s columns rotated by `rot`, flattened; `unit` normalizes each one first."
function _rotatedsample(rot, X::AbstractMatrix, samplesize::Int, rng; unit::Bool)
    n = size(X, 2)
    ids = rand(rng, 1:n, min(samplesize, n))
    dim = size(X, 1)
    V = Vector{Float32}(undef, dim * length(ids))
    for (j, i) in enumerate(ids)
        r = rotate(rot, view(X, :, i))
        if unit
            nr = sqrt(sum(abs2, r))
            nr > 0 && (r ./= nr)
        end
        copyto!(V, (j - 1) * dim + 1, r, 1, dim)
    end
    V
end

"Quantizes the (already rotated) vector `r` with the parameters `E`: the `SQVec` the storage receives."
function _quantcode(::Val{B}, E::SQMinC, r::AbstractVector{Float32}) where {B}
    codes = Vector{UInt8}(undef, cld(length(r), codesperbyte(Val(B))))
    SQVec{B}(E, packcodes!(Val(B), codes, r, E.min, 1f0 / E.c))
end

function _quantcode(::Val{B}, ::Nothing, r::AbstractVector{Float32}) where {B}
    codes = Vector{UInt8}(undef, cld(length(r), codesperbyte(Val(B))))
    SQVec{B}(quantvector!(Val(B), codes, r), codes)
end

"""
    SQEncoder(quant::Module, X::AbstractMatrix; dist=ScalarQuant.SqL2(), samplesize=4096, rng)
    SQEncoder(quant::Module, dim::Integer; dist=ScalarQuant.SqL2(), minmax=nothing)
    SQEncoder(quant::Module, rotation, X::AbstractMatrix; ...)
    SQEncoder(quant::Module, rotation, dim::Integer; ...)

The scalar quantizers as the encoder of an `AsymmetricSearchGraph`: an object is quantized
once (after an optional rotation) and stored as its codes, a raw query is kept in `Float32`
(rotated the same way), and this module's distances evaluate one against the other. It goes
through the `AbstractEstimator` interface -- `encode`, `encodequery`, `evaluate` -- because
that is what the graph navigates with, but it is a codification, not an estimate with an
error to exploit: nothing is bounded and nothing is ever re-evaluated.

`quant` is one of `ScalarQuant`'s quantizer modules, which names the family and the width at
once: `SQgu8`, `SQgu4`, `SQgu2` (one global range for every code, estimated by `sqautorange`
on a sample of `X`'s rotated coordinates, or given as `minmax`) or `SQu8`, `SQu4`, `SQu2`
(each code with its own range from its own extrema, which needs nothing beyond `dim`).

`rotation` is the object that rotates, a [`SimilaritySearch.Projections.Rotation`](@ref) -- `Projections.qr(dim, dim)`
or `Projections.RandomizedHadamard(dim)` -- or `nothing`, which quantizes the coordinates as
they are and is **the default**: the forms without a `rotation` argument rotate nothing. A
rotation gives every coordinate the same scale, which is what a single global range rests on;
on data whose coordinates already share one (normalized embeddings: on SISAP 2025 `ccnews` a
QR rotation moved recall@10 by less than 0.01 at every width) it changes nothing and costs its
flops per query and per inserted item. Rotate when the coordinates' scales are uneven and the
global family is wanted anyway; the per-vector family is the other remedy for uneven scales.

`encode` rotates an object once and quantizes the rotated vector, `encodequery` rotates the
query once and prepares it (an [`SQQuery`](@ref): the rotated `Float32` vector with its sums
and an integer image on its own range), and `evaluate` is `dist`, one of this module's
(`SqL2`, `L2`, `L1`, `NormCosine` or `Cosine`): its factored kernel for the prepared query
against a stored code, one integer dot product per pair, and its integer kernels between two
stored codes. Nothing is ever rotated or prepared inside an evaluation. [`sqcodes`](@ref)`(e)` is the storage to build an `AsymmetricSearchGraph`
over: a `QuantDatabase` with the estimator's own parameters, whose codes live in dense
blocks, and which takes the `SQVec`s `encode` produces without re-quantizing them.
"""
struct SQEncoder{B,ROT,P,D} <: AbstractEstimator
    rot::ROT
    dim::Int
    E::P              # SQMinC for the global family, nothing for the per-vector one
    dist::D
end

quantizer(e::SQEncoder{B}) where {B} = _quantmodule(B, isglobal(e))
codewidth(::SQEncoder{B}) where {B} = B
isglobal(e::SQEncoder) = e.E isa SQMinC

_checkrotation(::Nothing, ::Int) = true
_checkrotation(rot, dim::Int) = rotationdim(rot) == dim ||
    throw(ArgumentError("SQEncoder: a rotation of dimension $(rotationdim(rot)) for vectors of dimension $dim"))

function SQEncoder(quant::Module, rot, X::AbstractMatrix;
        dist=SqL2(), samplesize::Int=4096, rng::AbstractRNG=Random.default_rng())
    dim = size(X, 1)
    _checkrotation(rot, dim)
    B, E = _quantparams(quant, () -> _rotatedsample(rot, X, samplesize, rng; unit=false))
    SQEncoder{B,typeof(rot),typeof(E),typeof(dist)}(rot, dim, E, dist)
end

SQEncoder(quant::Module, X::AbstractMatrix; kwargs...) = SQEncoder(quant, nothing, X; kwargs...)
SQEncoder(quant::Module, dim::Integer; kwargs...) = SQEncoder(quant, nothing, dim; kwargs...)

function SQEncoder(quant::Module, rot, dim::Integer; dist=SqL2(), minmax=nothing)
    dim = Int(dim)
    _checkrotation(rot, dim)
    B, E = _quantparams(quant, minmax)
    SQEncoder{B,typeof(rot),typeof(E),typeof(dist)}(rot, dim, E, dist)
end

_checkdim(e::SQEncoder, v) = length(v) == e.dim ||
    throw(ArgumentError("SQEncoder: a vector of dimension $(length(v)) for an estimator of dimension $(e.dim)"))

"Rotates once and quantizes: the `SQVec` the storage receives."
function encode(e::SQEncoder{B}, o::AbstractVector) where {B}
    _checkdim(e, o)
    _quantcode(Val(B), e.E, rotate(e.rot, o))
end

"""
Rotates once and prepares the query for the mixed kernels: an [`SQQuery`](@ref) of the
encoder's width, with the sums and the integer image the factored distances take (issue
#110). It is still an `AbstractVector{Float32}`, the rotated query, for whatever reads the
coordinates.
"""
function encodequery(e::SQEncoder{B}, q::AbstractVector) where {B}
    _checkdim(e, q)
    SQQuery{B}(rotate(e.rot, q))
end

@inline evaluate(e::SQEncoder, q::AbstractVector{Float32}, s::SQVec)::Float32 = evaluate(e.dist, q, s)
@inline evaluate(e::SQEncoder, s::SQVec, q::AbstractVector{Float32})::Float32 = evaluate(e.dist, q, s)
@inline evaluate(e::SQEncoder, a::SQVec, b::SQVec)::Float32 = evaluate(e.dist, a, b)

"""
    sqcodes(e::SQEncoder) -> QuantDatabase

An empty, growable `QuantDatabase` with `e`'s own parameters and dense code blocks: the
storage of an `AsymmetricSearchGraph` over `e`. It takes the `SQVec`s `encode` produces as
they are.
"""
function sqcodes(e::SQEncoder{B}) where {B}
    nb = cld(e.dim, codesperbyte(Val(B)))
    Q = BlockMatrixDatabase(nb, UInt8)
    isglobal(e) ? QuantDatabase{B}(e.E, Q; dim=e.dim) : QuantDatabase{B}(SQMinC[], Q; dim=e.dim)
end

"""
    sqcodes(e::SQEncoder, X::AbstractMatrix; minbatch=4) -> QuantDatabase

Encodes every column of `X` in parallel into the storage above.
"""
function sqcodes(e::SQEncoder{B}, X::AbstractMatrix; minbatch::Int=4) where {B}
    n = size(X, 2)
    codes = Vector{SQVec{B,Vector{UInt8}}}(undef, n)
    @BATCHES minbatch for i in 1:n
        codes[i] = encode(e, view(X, :, i))
    end
    db = sqcodes(e)
    for c in codes
        push_item!(db, c)
    end
    db
end

function Base.show(io::IO, e::SQEncoder)
    print(io, "SQEncoder(", nameof(quantizer(e)), ", ", rotationname(e.rot), ", dim=", e.dim, ", dist=", e.dist, ")")
end
