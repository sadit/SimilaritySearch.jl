# This file is a part of SimilaritySearch.jl

"""
    SQQuery{B}(q::AbstractVector) <: AbstractVector{Float32}

A query prepared once for the mixed kernels against `SQVec{B}` codes (issue #110): what
[`SQEncoder`](@ref)'s `encodequery` returns, and what an `AsymmetricSearchGraph` evaluates
its distance with, once per query and once per inserted item.

With `x̂ᵢ = c·aᵢ + m` the squared Euclidean distance expands as

    Σ x̂ᵢ² − 2 Σ x̂ᵢ qᵢ + ‖q‖²  =  (c²·Saa + 2cm·Sa + n·m²)  −  2·(c·Σ aᵢqᵢ + m·Σqᵢ)  +  ‖q‖²

where the first term comes from the sums every `SQVec` stores, the last two from this
object, and only `Σ aᵢqᵢ` is evaluated per pair. That dot product is taken with an **integer
image** of the query: `qᵢ ≈ mq + sq·uᵢ` on the query's own range, `uᵢ` at 15 bits against
8-bit codes and at 8 bits against 4- and 2-bit codes, stored signed (`dᵢ = uᵢ − half`) in
the planes the packed codes unpack into, so `Σ aᵢqᵢ = mq·Sa + sq·(Σ aᵢdᵢ + half·Sa)`.
Measured on SISAP 2025 ccnews against the exact distance: the 8-bit image against 4-bit
codes is as accurate as the `Float32` query (mean error 0.0165 for both, against 0.0268 for a
query quantized on the codes' global range), and the 15-bit image against 8-bit codes is
within 1e-4 of it. The kernels are the integer dot products of `codes.jl`, the symmetric
kernels' cost.

It is still an `AbstractVector{Float32}` -- the rotated query, coordinate by coordinate --
so everything that takes a plain query keeps working on it: `L1`, the generic scans, a
`QuantDatabase` quantizing it for the symmetric graph. `length(q)` must be a multiple of
the codes per byte (2 at 4 bits, 4 at 2 bits), as for `SQVec`.
"""
struct SQQuery{B,T<:Union{Int8,Int16},P} <: AbstractVector{Float32}
    q::Vector{Float32}
    sumq::Float64           # Σ qᵢ
    sumqq::Float64          # Σ qᵢ²
    planes::NTuple{P,Vector{T}}   # the integer image, one plane per code field of a byte
    mq::Float32             # qᵢ ≈ mq + sq·(dᵢ + half)
    sq::Float32
    half::Int32
end

_querybits(::Val{8}) = 15
_querybits(::Val{4}) = 8
_querybits(::Val{2}) = 8
_querytype(::Val{8}) = Int16
_querytype(::Val{4}) = Int8
_querytype(::Val{2}) = Int8

function SQQuery{B}(q::AbstractVector{<:Real}) where {B}
    cpb = codesperbyte(Val(B))
    n = length(q)
    n % cpb == 0 ||
        throw(ArgumentError("SQQuery{$B}: length(q) = $n must be a multiple of $cpb ($cpb coordinates are packed per UInt8)"))
    qf = q isa Vector{Float32} ? q : Vector{Float32}(q)
    bits = _querybits(Val(B)); T = _querytype(Val(B))
    lv = (1 << bits) - 1; half = 1 << (bits - 1)
    lo, hi = extrema(qf)
    sq = hi > lo ? (hi - lo) / Float32(lv) : 1f0
    sumq = 0.0; sumqq = 0.0
    @inbounds for i in 1:n
        x = Float64(qf[i]); sumq += x; sumqq += x * x
    end
    # plane p holds the coordinates a byte's field p unpacks into: coordinate cpb*(i-1) + p + 1 for byte i
    planes = ntuple(p -> T[T(clamp(round(Int, (qf[cpb * (i - 1) + p] - lo) / sq), 0, lv) - half) for i in 1:(n ÷ cpb)], cpb)
    SQQuery{B,T,cpb}(qf, sumq, sumqq, planes, lo, sq, Int32(half))
end

Base.size(Q::SQQuery) = size(Q.q)
Base.IndexStyle(::Type{<:SQQuery}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(Q::SQQuery, i::Integer) = Q.q[i]
codewidth(::SQQuery{B}) where {B} = B

function Base.show(io::IO, Q::SQQuery{B,T}) where {B,T}
    print(io, "SQQuery{", B, "}(dim=", length(Q.q), ", image=", T, ", ‖q‖=", round(sqrt(Q.sumqq), digits=4), ")")
end
Base.show(io::IO, ::MIME"text/plain", Q::SQQuery) = show(io, Q)

"Σ aᵢqᵢ between the codes of `A` and the query `Q`, from the integer image -- see [`SQQuery`](@ref)."
@inline function codequerydot(A::SQVec{B}, Q::SQQuery{B})::Float64 where {B}
    s = Float64(dotquery(Val(B), A.V, Q.planes...))
    Float64(Q.mq) * Float64(A.Sa) + Float64(Q.sq) * (s + Float64(Q.half) * Float64(A.Sa))
end
