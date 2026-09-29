"""
    SQu8

Per-vector (per-column) 8-bit scalar quantization: [`quantize`](@ref SQu8.quantize) stores
one `UInt8` code per coordinate, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu8.quantize`, etc.

The vector type and the distances are the module-wide [`SQVec`](@ref ScalarQuant.SQVec)
and [`SqL2`](@ref ScalarQuant.SqL2)/[`L2`](@ref ScalarQuant.L2)/[`L1`](@ref ScalarQuant.L1)/
[`NormCosine`](@ref ScalarQuant.NormCosine); the names here are aliases kept for the
per-width API. See also [`SQgu8`](@ref ScalarQuant.SQgu8)
for a variant that shares a single pair of quantization parameters across all columns.
"""
module SQu8

export quantize, SQu8Vec, SQu8Database, L1, L2, SqL2, NormCosine

using ..ScalarQuant: SQMinC, AbstractDatabase, getminbatch, @BATCHES, SQVec, codesums, quantvector!
using ..ScalarQuant: L1, L2, SqL2, NormCosine

"8-bit [`SQVec`](@ref ScalarQuant.SQVec), one `UInt8` code per coordinate. `SQu8Vec(v)` quantizes `v` on its own extrema."
const SQu8Vec = SQVec{8}

"""
    quantize(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 8 bits per coordinate (one `UInt8` per
coordinate). This reduces the memory footprint of a database of vectors by roughly a
factor of 4 with respect to `Float32` at the cost of precision. Each column is quantized
independently using its own minimum and scale factor, computed from the extrema of the
column so that the whole range `[min, max]` is mapped to the codes `\\{0, 1, \\ldots, 255\\}`.

`quantize` creates a `SQu8Database` struct that follows the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu8Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

See also [`SQgu8`](@ref ScalarQuant.SQgu8)'s `quantize` for a variant that shares a single pair of
quantization parameters across all columns instead of computing them per column.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
function quantize(X::AbstractMatrix)
    SQu8Database(X)
end

"""
    SQu8Database(X::AbstractMatrix)
    SQu8Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})

An [`AbstractDatabase`](@ref) of vectors quantized to 8 bits per coordinate, one `UInt8` per coordinate,
each column carrying its **own** `min`/scale pair (`E[i]`) computed from that vector's own
extrema. Indexing yields a [`SQu8Vec`](@ref), which the distances in this module consume
without dequantizing.

The first constructor quantizes `X` (it is what [`quantize`](@ref) calls). The second takes the
two fields back as they are, quantizing nothing -- that is the one to use after reading `E`/`Q`
from storage, so a stored database goes straight back to work without rebuilding the `Float32`
matrix it came from (which would cost 4x the memory the quantization was chosen to avoid, to
recompute codes that are already in hand). The two must agree: exactly one `SQMinC` per stored
vector, i.e. `length(E) == size(Q, 2)`.

# Fields
- `E::Vector{SQMinC}`: per-column `min`/scale, one entry per stored vector
- `Q::Matrix{UInt8}`: the codes, `size(X, 1)` rows by one column per vector
"""
struct SQu8Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}
    Sa::Vector{Float32}      # per column: Σ codes, Σ codes² -- see `codesums`. Derived from
    Saa::Vector{Float32}     # `Q` alone, so they are recomputed rather than stored/read.

    function SQu8Database(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, m, n)
        E = Vector{SQMinC}(undef, n)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            E[i] = quantvector!(Val(8), view(Q, :, i), view(X, :, i))
            Sa[i], Saa[i] = codesums(Val(8), view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end

    function SQu8Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})
        length(E) == size(Q, 2) ||
            throw(ArgumentError("SQu8Database: got $(length(E)) quantization parameters for $(size(Q, 2)) columns; there is exactly one `SQMinC` per stored vector"))
        n = size(Q, 2)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            Sa[i], Saa[i] = codesums(Val(8), view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end
end

Base.eltype(Q::SQu8Database) = typeof(Q[1])
Base.length(Q::SQu8Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu8Database, i::Integer)
   SQu8Vec(Q.E[i], view(Q.Q, :, i), Q.Sa[i], Q.Saa[i])
end

"""
    quantize(db::SQu8Database, v::AbstractVector)

Quantizes a single vector `v` to 8 bits per coordinate, the same way as the vectors
already stored in `db`, returning a [`SQu8Vec`](@ref). Since [`SQu8`](@ref) computes each
vector's own `min`/scale independently from its own extrema (see [`quantize(X::AbstractMatrix)`](@ref)),
this does not read or depend on `db`'s stored data or parameters; `db` is only used to
validate that `v` has the expected dimension. This is convenient, e.g., to quantize a
query vector the same way as the vectors stored in `db`, so that it can be compared
against them with [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref)/[`NormCosine`](@ref).

# Arguments
- `db`: the database `v` should be dimensionally consistent with
- `v`: the vector to quantize; `length(v)` must equal `db`'s vector dimension

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8.quantize(X);

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQu8.quantize(db, q);  # quantized the same way as db's vectors
```
"""
function quantize(db::SQu8Database, v::AbstractVector)
    expected = size(db.Q, 1)
    length(v) == expected || throw(ArgumentError("SQu8.quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    SQu8Vec(v)
end


end
