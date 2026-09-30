# This file is a part of SimilaritySearch.jl

export RaBitQRefined, RaBitQExactFallback, RaBitQVectorFallback, refinethreshold


"""
    abstract type AbstractFallback end

The finer representation a [`RaBitQRefined`](@ref) keeps beside the bits, in the rotated
space: `finecode(f, r)` builds it from the rotated vector `r`, and `finecos(f, fine, qr)` is
the cosine between it and the rotated unit query `qr`. Two are provided: [`RaBitQExactFallback`](@ref)
and [`RaBitQVectorFallback`](@ref).
"""
abstract type AbstractFallback end

"""
    RaBitQExactFallback{T}()   (T = Float32 by default, or Float16)

The rotated vector itself, normalized, in `T`: `4·dim` (`2·dim`) bytes per object and the
exact cosine when it is consulted. This is what RaBitQ's own search re-ranks with, and the
case where the bits pay: navigation costs the bit estimate, and the exact evaluation is only
paid for the candidates the error bound cannot rule out.
"""
struct RaBitQExactFallback{T<:AbstractFloat} <: AbstractFallback end
RaBitQExactFallback() = RaBitQExactFallback{Float32}()

function finecode(::RaBitQExactFallback{T}, r::AbstractVector{Float32}) where {T}
    nr = sqrt(sum(abs2, r))
    nr > 0 ? Vector{T}(r ./ nr) : Vector{T}(r)
end

@inline function finecos(::RaBitQExactFallback, fine::AbstractVector, qr::AbstractVector{Float32})::Float32
    d = 0f0
    @inbounds @simd for i in eachindex(qr)
        d += Float32(fine[i]) * qr[i]
    end
    d
end

"""
    RaBitQVectorFallback(quant::Module, est::AbstractRaBitQ, X::AbstractMatrix; samplesize=4096, rng)
    RaBitQVectorFallback(quant::Module, est::AbstractRaBitQ; minmax=nothing)

The rotated unit vector scalar-quantized with `quant`, one of `ScalarQuant`'s quantizer
modules, which names the family and the width at once: `SQgu8`, `SQgu4`, `SQgu2` (one global
range for every code, estimated by `sqautorange` on a sample of `X`'s rotated unit vectors,
or given as `minmax`) or `SQu8`, `SQu4`, `SQu2` (each code with its own range, which needs no
data). `est` supplies the rotation and the dimension. `dim·B/8 + 8` bytes per object,
consulted through `ScalarQuant`'s mixed kernels with the dequantized norm.
"""
struct RaBitQVectorFallback{B,P} <: AbstractFallback
    E::P              # SQMinC for the global family, nothing for the per-vector one
    dim::Int
end

function RaBitQVectorFallback(quant::Module, est::AbstractRaBitQ, X::AbstractMatrix; samplesize::Int=4096, rng::AbstractRNG=Random.default_rng())
    B, E = _quantparams(quant, () -> _rotatedsample(est.rot, X, samplesize, rng; unit=true))
    RaBitQVectorFallback{B,typeof(E)}(E, est.dim)
end

function RaBitQVectorFallback(quant::Module, est::AbstractRaBitQ; minmax=nothing)
    B, E = _quantparams(quant, minmax)
    RaBitQVectorFallback{B,typeof(E)}(E, est.dim)
end

codewidth(::RaBitQVectorFallback{B}) where {B} = B
isglobal(f::RaBitQVectorFallback) = f.E isa SQMinC
quantizer(f::RaBitQVectorFallback{B}) where {B} = _quantmodule(B, isglobal(f))

function finecode(f::RaBitQVectorFallback{B}, r::AbstractVector{Float32}) where {B}
    nr = sqrt(sum(abs2, r))
    _quantcode(Val(B), f.E, nr > 0 ? r ./ nr : r)
end

@inline finecos(::RaBitQVectorFallback, fine::SQVec, qr::AbstractVector{Float32})::Float32 =
    Float32(dotmixed(fine, qr) / max(quantnorm(fine), 1e-12))

"""
    RaBitQRefined(coarse::RaBitQCosine, fine::AbstractFallback; τ=Inf32)

Two levels in one estimator: the bits of `coarse` navigate, and the finer representation
beside them re-evaluates, inside the same `evaluate`, every object whose estimated cosine
distance minus its error bound is at or below `τ` -- the objects the bits cannot rule out as
far. `τ = Inf` re-evaluates everything (the fine level decides every distance); a finite `τ`
on the scale of the k-th neighbor's distance, which [`refinethreshold`](@ref) reads off a
sample, keeps the fine evaluations to the candidates that can matter. Between two stored
objects (the graph's neighborhood filters) only the bits are compared.

The stored object is a tuple `(RaBitQCode, fine)`; [`rabitqcodes`](@ref)`(e)` gives an empty
growable storage of the right element type.
"""
struct RaBitQRefined{C<:RaBitQCosine,F<:AbstractFallback} <: AbstractEstimator
    coarse::C
    fine::F
    τ::Float32
end

RaBitQRefined(coarse::RaBitQCosine, fine::AbstractFallback; τ::Real=Inf32) = RaBitQRefined(coarse, fine, Float32(τ))

function encode(e::RaBitQRefined, o::AbstractVector)
    _checkdim(e.coarse, o)
    r = rotate(e.coarse.rot, o)
    (_encode_rotated(e.coarse, r, Float32(sqrt(sum(abs2, o)))), finecode(e.fine, r))
end

encodequery(e::RaBitQRefined, q::AbstractVector) = encodequery(e.coarse, q)

@inline function evaluate(e::RaBitQRefined, q::RaBitQQuery, s::Tuple)::Float32
    code = s[1]
    d = 1f0 - estimatecos(e.coarse, q, code)
    d - code.err <= e.τ ? 1f0 - finecos(e.fine, s[2], q.r) : d
end

@inline evaluate(e::RaBitQRefined, a::Tuple, b::Tuple)::Float32 = evaluate(e.coarse, a[1], b[1])
@inline evaluate(e::RaBitQRefined, s::Tuple, q::RaBitQQuery) = evaluate(e, q, s)

rabitqcodes(e::RaBitQRefined) = VectorDatabase(type=typeof(encode(e, zeros(Float32, e.coarse.dim))))

function rabitqcodes(e::RaBitQRefined, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)
    codes = Vector{typeof(encode(e, view(X, :, 1)))}(undef, n)
    @BATCHES minbatch for i in 1:n
        codes[i] = encode(e, view(X, :, i))
    end
    VectorDatabase(codes)
end

"""
    refinethreshold(est::AbstractRaBitQ, X, k; samplesize=2000, numqueries=200, q=0.9, rng) -> Float32

A `τ` for [`RaBitQRefined`](@ref) on the scale of the data: the `q`-quantile, over
`numqueries` objects of `X`, of the exact cosine distance to their `k`-th nearest neighbor
within a sample of `samplesize` objects. A candidate whose estimated distance minus its
error bound is within that is one the search may keep, and is worth the fine evaluation.
"""
function refinethreshold(est::AbstractRaBitQ, X::AbstractMatrix, k::Integer;
        samplesize::Int=2000, numqueries::Int=200, q::Real=0.9, rng::AbstractRNG=Random.default_rng())
    n = size(X, 2)
    ids = rand(rng, 1:n, min(samplesize, n))
    S = Matrix{Float32}(undef, size(X, 1), length(ids))
    for (j, i) in enumerate(ids)
        S[:, j] .= view(X, :, i) ./ sqrt(sum(abs2, view(X, :, i)))
    end
    seq = ExhaustiveSearch(Dist.NormCosine(), MatrixDatabase(S))
    ctx = GenericContext()
    kth = Float32[]
    for j in rand(rng, 1:length(ids), min(numqueries, length(ids)))
        res = knnqueue(KnnSorted, k + 1)                     # the object itself is in the sample
        search(seq, ctx, view(S, :, j), res)
        push!(kth, maximum(res))
    end
    Float32(quantile(kth, q))
end

Base.show(io::IO, f::RaBitQVectorFallback) = print(io, "RaBitQVectorFallback(", nameof(quantizer(f)), ", dim=", f.dim, ")")

function Base.show(io::IO, e::RaBitQRefined)
    print(io, "RaBitQRefined(", e.coarse, ", ", e.fine, ", τ=", e.τ, ")")
end
