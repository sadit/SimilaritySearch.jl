# This file is a part of SimilaritySearch.jl

export SQVec, SQCodes

"""
    quantvector!(::Val{B}, vout::AbstractVector{UInt8}, v::AbstractVector; range=DEFAULT_RANGE, eps=1f-6) -> SQMinC

Quantizes `v` into `vout` on its **own** range, placed by `range` (a [`RangePolicy`](@ref);
the default, [`AutoRange`](@ref), resolves by width), so a vector on a different scale from the rest
is encoded as faithfully as any other. Returns the [`SQMinC`](@ref) that dequantizes it; `eps`
keeps a constant vector's range from collapsing.
"""
function quantvector!(B::Val, vout::AbstractVector{UInt8}, v::AbstractVector; range::RangePolicy=DEFAULT_RANGE, eps::Float32=1f-6)
    min, c = vectorrange(range, v, levels(B); eps)
    packcodes!(B, vout, v, min, 1f0/c)
    SQMinC(min, c)
end

"""
    SQVec{B,VEC<:AbstractVector{UInt8}}

    SQVec{B}(v::AbstractVector)
    SQVec{B}(E::SQMinC, V::AbstractVector{UInt8})
    SQVec{B}(E::SQMinC, V::AbstractVector{UInt8}, Sa, Saa)

A single vector quantized to `B` bits per coordinate (2, 4 or 8): the packed codes `V`
(`codesperbyte(B)` coordinates to a byte, low bits first), the affine dequantization
parameters `E` (coordinate `i` is `code * E.c + E.min`), and the two code sums `Sa = Σ codes`
and `Saa = Σ codes²` that let every distance in this module be computed from one integer
pass over the codes (see the note above `codesums` in `codes.jl`). Indexing (`qvec[i]`)
dequantizes coordinate `i` to a `Float32`.

It is the element type of every quantized database here, in both families: a globally
quantized vector *is* a per-vector one whose `E` happens to be shared with the rest of its
database. `SQu8Vec`, `SQu4Vec` and `SQu2Vec` are aliases for `SQVec{8}`, `SQVec{4}` and
`SQVec{2}`.

The first constructor quantizes `v` on its own range ([`quantvector!`](@ref), placed by `range`). `length(v)`
must be a multiple of `codesperbyte(B)` (2 at 4 bits, 4 at 2 bits), or an `ArgumentError`
is thrown: pad `v` if needed, and then pad any plain vector later compared against the
result the same way, since the mixed distances index it positionally. The other two take
stored codes back as they are, recomputing the sums from the codes when they are not given.
"""
struct SQVec{B,VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
    Sa::Float32      # Σ codes      -- see the expansion above `codesums`
    Saa::Float32     # Σ codes²
end

SQVec{B}(E::SQMinC, V::VEC, Sa::Real, Saa::Real) where {B,VEC<:AbstractVector{UInt8}} =
    SQVec{B,VEC}(E, V, Float32(Sa), Float32(Saa))

SQVec{B}(E::SQMinC, V::AbstractVector{UInt8}) where {B} = SQVec{B}(E, V, codesums(Val(B), V)...)

function SQVec{B}(v::AbstractVector; range::RangePolicy=DEFAULT_RANGE) where {B}
    cpb = codesperbyte(Val(B))
    length(v) % cpb == 0 ||
        throw(ArgumentError("SQVec{$B}: length(v) = $(length(v)) must be a multiple of $cpb ($cpb coordinates are packed per UInt8)"))
    vout = Vector{UInt8}(undef, length(v) ÷ cpb)
    E = quantvector!(Val(B), vout, v; range)
    SQVec{B}(E, vout)
end

"The code width of `q`, in bits."
codewidth(::SQVec{B}) where {B} = B

Base.@propagate_inbounds function Base.getindex(q::SQVec{B}, i::Integer)::Float32 where {B}
    Float32(getcode(Val(B), q.V, i)) * q.E.c + q.E.min
end

Base.length(q::SQVec{B}) where {B} = codesperbyte(Val(B)) * length(q.V)
Base.eachindex(q::SQVec) = 1:length(q)
Base.eltype(::SQVec) = Float32
Base.eltype(::Type{<:SQVec}) = Float32

"""
    SQCodes{B,VEC<:AbstractVector{UInt8}}

    SQCodes{B}(E::SQMinC, V::AbstractVector{UInt8})

A vector quantized to `B` bits per coordinate without its code sums: the packed codes `V` and
the parameters `E` that dequantize them, nothing else. It is what a
[`GlobalCodesDatabase`](@ref) yields, and it exists because the distances between two codes of
the same scale -- `SqL2`, `L2` and `L1` in the global family -- are one integer pass over the
codes and never read `Sa`/`Saa`. An [`SQVec`](@ref) still loads both from the database for every
evaluation, two more memory accesses per child in a graph search (measured below in
[`GlobalCodesDatabase`](@ref)).

Every distance of this module takes it. Where the sums are needed (codes of different scales,
`NormCosine`, `Cosine`, a prepared [`SQQuery`](@ref)) they are computed from the codes on the
spot, one more pass per evaluation, and the result is the very value the `SQVec` gives; a plain
`Float32` vector is scanned as it is against an `SQVec`. `SQVec(a)` adds the sums back.
"""
struct SQCodes{B,VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
end

SQCodes{B}(E::SQMinC, V::VEC) where {B,VEC<:AbstractVector{UInt8}} = SQCodes{B,VEC}(E, V)
SQVec(a::SQCodes{B}) where {B} = SQVec{B}(a.E, a.V)

codewidth(::SQCodes{B}) where {B} = B

Base.@propagate_inbounds function Base.getindex(q::SQCodes{B}, i::Integer)::Float32 where {B}
    Float32(getcode(Val(B), q.V, i)) * q.E.c + q.E.min
end

Base.length(q::SQCodes{B}) where {B} = codesperbyte(Val(B)) * length(q.V)
Base.eachindex(q::SQCodes) = 1:length(q)
Base.eltype(::SQCodes) = Float32
Base.eltype(::Type{<:SQCodes}) = Float32

# a quantized vector with or without its sums: what the kernels that read only `E` and `V` take
const _SQ{B} = Union{SQVec{B},SQCodes{B}}

"`a` with its code sums: itself for an `SQVec`, the sums computed from the codes for an `SQCodes`."
@inline _withsums(a::SQVec) = a
@inline _withsums(a::SQCodes{B}) where {B} = SQVec{B}(a.E, a.V)
