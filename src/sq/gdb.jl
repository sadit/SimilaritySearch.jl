# This file is a part of SimilaritySearch.jl

export GlobalQuantDatabase

"""
    GlobalQuantDatabase{B}

    GlobalQuantDatabase(bits::Integer, X::AbstractMatrix; minmax=nothing, storage=MatrixDatabase, quant=nothing, samplesize=0)
    GlobalQuantDatabase(bits::Integer, Q, minmax; dim=nothing, Sa=nothing, Saa=nothing)

The [`QuantDatabase`](@ref) of vectors quantized to `bits` bits per coordinate (2, 4 or 8)
under **one** `min`/scale pair shared by the whole dataset, keeping that pair -- and the
per-vector code sums -- alongside the codes.

That is the difference from calling [`SQgu8.quantize`](@ref SimilaritySearch.ScalarQuant.SQgu8.quantize)
directly, which hands back a bare `Matrix{UInt8}` and leaves `minmax` to the caller. Without
the parameters a stored matrix of codes cannot be dequantized at all, so it can only ever be
compared against other codes from the same run; with them, a query may stay in its original
`Float32` form, and a vector arriving later is quantized with the very same range
([`push_item!`](@ref QuantDatabase)), which is what keeps its codes comparable.

Indexing yields an [`SQVec`](@ref) of the same width the per-vector quantizers produce -- a
globally quantized vector *is* a per-vector one whose scale happens to be shared -- so every
distance in this module works here unchanged, and each takes the path it should:

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
- `Q`, `minmax`: already-quantized codes (a matrix, or any `AbstractDatabase` of `UInt8`
  vectors, an empty one to start a growable database) and the pair they were produced with

# Keyword Arguments
- `minmax`: the `(min, max)` pair to quantize with; estimated from a sample of `X` by
  [`sqrange`](@ref) when not given, exactly as the underlying `quantize` does
- `storage`: the database the codes go to, as a function of their `Matrix{UInt8}`
- `dim`, `Sa`, `Saa`: see [`QuantDatabase`](@ref)
"""
const GlobalQuantDatabase = QuantDatabase{B,SQMinC} where {B}

function _checkbits(bits)
    bits in (2, 4, 8) || throw(ArgumentError("GlobalQuantDatabase: bits=$bits must be 2, 4 or 8"))
    Int(bits)
end

(::Type{GlobalQuantDatabase})(bits::Integer, X::AbstractMatrix; kwargs...) =
    QuantDatabase{_checkbits(bits),SQMinC}(X; kwargs...)

(::Type{GlobalQuantDatabase})(bits::Integer, Q::Union{AbstractMatrix{UInt8},AbstractDatabase}, minmax; kwargs...) =
    QuantDatabase{_checkbits(bits),SQMinC}(Q, minmax; kwargs...)
