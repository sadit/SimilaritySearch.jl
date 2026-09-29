"""
    SQu2

Per-vector (per-column) 2-bit scalar quantization: [`quantize`](@ref SQu2.quantize) packs
four 2-bit codes per `UInt8`, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu2.quantize`, etc.

The vector type and the distances are the module-wide [`SQVec`](@ref ScalarQuant.SQVec)
and [`SqL2`](@ref ScalarQuant.SqL2)/[`L2`](@ref ScalarQuant.L2)/[`L1`](@ref ScalarQuant.L1)/
[`NormCosine`](@ref ScalarQuant.NormCosine); the names here are aliases kept for the
per-width API.
"""
module SQu2

export quantize, SQu2Vec, SQu2Database, L1, L2, SqL2, NormCosine

using ..ScalarQuant: SQMinC, AbstractDatabase, getminbatch, @BATCHES, SQVec, codesums, quantvector!
using ..ScalarQuant: L1, L2, SqL2, NormCosine

"2-bit [`SQVec`](@ref ScalarQuant.SQVec), four 2-bit codes packed per `UInt8`. `SQu2Vec(v)` quantizes `v` on its own extrema."
const SQu2Vec = SQVec{2}

"""
    quantize(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 2 bits per coordinate, packing four
codes into each `UInt8`. This reduces the memory footprint of a database of vectors by
roughly a factor of 16 with respect to `Float32` at the cost of precision. Each column
is quantized independently using its own minimum and scale factor, computed from the
extrema of the column so that the whole range `[min, max]` is mapped to the `\\{0,1,2,3\\}`
codes.

`quantize` wraps `SQu2Database` that implements the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu2Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized; `size(X, 1)` (the
  dimension) must be a multiple of `4` (throws `ArgumentError` otherwise), since 4
  coordinates are packed into each `UInt8`. Pad `X` with extra rows to the next multiple
  of 4 if needed.

!!! note
    If `X` needs padding, any plain (non-quantized) query vectors later compared against
    the resulting database via [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref) must be padded to
    that same (padded) dimension too, since those distances index the plain vector
    positionally and do not know about the padding.

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu2.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
function quantize(X::AbstractMatrix)
    m, n = size(X)
    m % 4 == 0 || throw(ArgumentError("SQu2.quantize: size(X, 1) = $m must be a multiple of 4 (4 coordinates are packed per UInt8)"))
    Q = Matrix{UInt8}(undef, m ÷ 4, n)
    E = Vector{SQMinC}(undef, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        E[i] = quantvector!(Val(2), view(Q, :, i), view(X, :, i))
    end

    SQu2Database(E, Q)
end

struct SQu2Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}
    Sa::Vector{Float32}      # per column: Σ codes, Σ codes² -- derived from `Q` alone, so
    Saa::Vector{Float32}     # they are recomputed rather than stored or read back

    """
        SQu2Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})

    Rebuilds a database from its stored fields, quantizing nothing; the code sums are
    recomputed from `Q` in one pass, so nothing but `E` and `Q` has to be persisted.
    """
    function SQu2Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})
        length(E) == size(Q, 2) ||
            throw(ArgumentError("SQu2Database: got $(length(E)) quantization parameters for $(size(Q, 2)) columns; there is exactly one `SQMinC` per stored vector"))
        n = size(Q, 2)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            Sa[i], Saa[i] = codesums(Val(2), view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end
end

Base.eltype(Q::SQu2Database) = typeof(Q[1])
Base.length(Q::SQu2Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu2Database, i::Integer)
   SQu2Vec(Q.E[i], view(Q.Q, :, i), Q.Sa[i], Q.Saa[i])
end

"""
    quantize(db::SQu2Database, v::AbstractVector)

Quantizes a single vector `v` to 2 bits per coordinate, the same way as the vectors
already stored in `db`, returning a [`SQu2Vec`](@ref). Since [`SQu2`](@ref) computes each
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

julia> db = ScalarQuant.SQu2.quantize(X);

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQu2.quantize(db, q);  # quantized the same way as db's vectors
```
"""
function quantize(db::SQu2Database, v::AbstractVector)
    expected = 4size(db.Q, 1)
    length(v) == expected || throw(ArgumentError("SQu2.quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    SQu2Vec(v)
end


end
