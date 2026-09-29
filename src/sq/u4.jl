"""
    SQu4

Per-vector (per-column) 4-bit scalar quantization: [`quantize`](@ref SQu4.quantize) packs
two 4-bit codes per `UInt8`, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu4.quantize`, etc.

The vector type and the distances are the module-wide [`SQVec`](@ref ScalarQuant.SQVec)
and [`SqL2`](@ref ScalarQuant.SqL2)/[`L2`](@ref ScalarQuant.L2)/[`L1`](@ref ScalarQuant.L1)/
[`NormCosine`](@ref ScalarQuant.NormCosine); the names here are aliases kept for the
per-width API.
"""
module SQu4

export quantize, SQu4Vec, SQu4Database, L1, L2, SqL2, NormCosine

using ..ScalarQuant: SQMinC, AbstractDatabase, getminbatch, @BATCHES, SQVec, codesums, quantvector!
using ..ScalarQuant: L1, L2, SqL2, NormCosine

"4-bit [`SQVec`](@ref ScalarQuant.SQVec), two 4-bit codes packed per `UInt8`. `SQu4Vec(v)` quantizes `v` on its own extrema."
const SQu4Vec = SQVec{4}

"""
    quantize(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 4 bits per coordinate, packing two
codes into each `UInt8`. This reduces the memory footprint of a database of vectors by
roughly a factor of 8 with respect to `Float32` at the cost of precision. Each column
is quantized independently using its own minimum and scale factor, computed from the
extrema of the column so that the whole range `[min, max]` is mapped to the codes
`\\{0, 1, \\ldots, 15\\}`.

`quantize` wraps `SQu4Database` that implements the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu4Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized; `size(X, 1)` (the
  dimension) must be a multiple of `2` (throws `ArgumentError` otherwise), since 2
  coordinates are packed into each `UInt8`. Pad `X` with an extra row if needed.

!!! note
    If `X` needs padding, any plain (non-quantized) query vectors later compared against
    the resulting database via [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref) must be padded to
    that same (padded) dimension too, since those distances index the plain vector
    positionally and do not know about the padding.

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu4.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
function quantize(X::AbstractMatrix)
    SQu4Database(X)
end

"""
    SQu4Database(X::AbstractMatrix)
    SQu4Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})

An [`AbstractDatabase`](@ref) of vectors quantized to 4 bits per coordinate, two 4-bit codes packed per `UInt8`,
each column carrying its **own** `min`/scale pair (`E[i]`) computed from that vector's own
extrema. Indexing yields a [`SQu4Vec`](@ref), which the distances in this module consume
without dequantizing.

The first constructor quantizes `X` (it is what [`quantize`](@ref) calls). The second takes the
two fields back as they are, quantizing nothing -- that is the one to use after reading `E`/`Q`
from storage, so a stored database goes straight back to work without rebuilding the `Float32`
matrix it came from (which would cost 8x the memory the quantization was chosen to avoid, to
recompute codes that are already in hand). The two must agree: exactly one `SQMinC` per stored
vector, i.e. `length(E) == size(Q, 2)`.

# Fields
- `E::Vector{SQMinC}`: per-column `min`/scale, one entry per stored vector
- `Q::Matrix{UInt8}`: the codes, `size(X, 1) ÷ 2` rows by one column per vector
"""
struct SQu4Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}
    Sa::Vector{Float32}      # per column: Σ codes, Σ codes² -- derived from `Q` alone, so
    Saa::Vector{Float32}     # they are recomputed rather than stored or read back

    function SQu4Database(X::AbstractMatrix)
        m, n = size(X)
        m % 2 == 0 || throw(ArgumentError("SQu4.quantize: size(X, 1) = $m must be a multiple of 2 (2 coordinates are packed per UInt8)"))
        Q = Matrix{UInt8}(undef, m ÷ 2, n)
        E = Vector{SQMinC}(undef, n)
        minbatch = getminbatch(n)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        @BATCHES minbatch for i in 1:n
            E[i] = quantvector!(Val(4), view(Q, :, i), view(X, :, i))
            Sa[i], Saa[i] = codesums(Val(4), view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end

    function SQu4Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})
        length(E) == size(Q, 2) ||
            throw(ArgumentError("SQu4Database: got $(length(E)) quantization parameters for $(size(Q, 2)) columns; there is exactly one `SQMinC` per stored vector"))
        n = size(Q, 2)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            Sa[i], Saa[i] = codesums(Val(4), view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end
end

Base.eltype(Q::SQu4Database) = typeof(Q[1])
Base.length(Q::SQu4Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu4Database, i::Integer)
   SQu4Vec(Q.E[i], view(Q.Q, :, i), Q.Sa[i], Q.Saa[i])
end

"""
    quantize(db::SQu4Database, v::AbstractVector)

Quantizes a single vector `v` to 4 bits per coordinate, the same way as the vectors
already stored in `db`, returning a [`SQu4Vec`](@ref). Since [`SQu4`](@ref) computes each
vector's own `min`/scale independently from its own extrema (see [`quantize(X::AbstractMatrix)`](@ref)),
this does not read or depend on `db`'s stored data or parameters; `db` is only used to
validate that `v` has the expected (padded) dimension. This is convenient, e.g., to
quantize a query vector the same way as the vectors stored in `db`, so that it can be
compared against them with [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref).

# Arguments
- `db`: the database `v` should be dimensionally consistent with
- `v`: the vector to quantize; `length(v)` must equal `db`'s (padded) vector dimension

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu4.quantize(X);

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQu4.quantize(db, q);  # quantized the same way as db's vectors
```
"""
function quantize(db::SQu4Database, v::AbstractVector)
    expected = 2size(db.Q, 1)
    length(v) == expected || throw(ArgumentError("SQu4.quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    SQu4Vec(v)
end


end
