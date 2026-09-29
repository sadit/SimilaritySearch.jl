# This file is a part of SimilaritySearch.jl

export SQVec

"""
    quantvector!(::Val{B}, vout::AbstractVector{UInt8}, v::AbstractVector; eps=1f-6) -> SQMinC

Quantizes `v` into `vout` on its **own** extrema, the per-vector family's rule: the whole
range `[min, max]` maps onto the codes `0:levels(B)`, so nothing is clipped and a vector on
a different scale from the rest is encoded as faithfully as any other. Returns the
[`SQMinC`](@ref) that dequantizes it; `eps` keeps a constant vector's range from collapsing.
"""
function quantvector!(B::Val, vout::AbstractVector{UInt8}, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / Float32(levels(B))
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

The first constructor quantizes `v` on its own extrema ([`quantvector!`](@ref)). `length(v)`
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

function SQVec{B}(v::AbstractVector) where {B}
    cpb = codesperbyte(Val(B))
    length(v) % cpb == 0 ||
        throw(ArgumentError("SQVec{$B}: length(v) = $(length(v)) must be a multiple of $cpb ($cpb coordinates are packed per UInt8)"))
    vout = Vector{UInt8}(undef, length(v) ÷ cpb)
    E = quantvector!(Val(B), vout, v)
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
