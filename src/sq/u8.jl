"""
    SQu8

Per-vector (per-column) 8-bit scalar quantization: [`quantize`](@ref SimilaritySearch.ScalarQuant.SQu8.quantize) stores
one `UInt8` code per coordinate, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu8.quantize`, etc. See also [`SQgu8`](@ref SimilaritySearch.ScalarQuant.SQgu8)
for a variant that shares a single pair of quantization parameters across all columns.

The vector type, the database and the distances are the module-wide
[`SQVec`](@ref SimilaritySearch.ScalarQuant.SQVec), [`QuantDatabase`](@ref SimilaritySearch.ScalarQuant.QuantDatabase) and
[`SqL2`](@ref SimilaritySearch.ScalarQuant.SqL2)/[`L2`](@ref SimilaritySearch.ScalarQuant.L2)/[`L1`](@ref SimilaritySearch.ScalarQuant.L1)/
[`NormCosine`](@ref SimilaritySearch.ScalarQuant.NormCosine); the names here are aliases kept for the
per-width API.
"""
module SQu8

export quantize, SQu8Vec, SQu8Database, L1, L2, SqL2, NormCosine

import ..ScalarQuant
using ..ScalarQuant: SQMinC, SQVec, QuantDatabase, L1, L2, SqL2, NormCosine

"8-bit [`SQVec`](@ref SimilaritySearch.ScalarQuant.SQVec), one `UInt8` code per coordinate. `SQu8Vec(v)` quantizes `v` on its own extrema."
const SQu8Vec = SQVec{8}

"""
    SQu8Database(X::AbstractMatrix; storage=MatrixDatabase)
    SQu8Database(E::AbstractVector{SQMinC}, Q; dim=nothing, Sa=nothing, Saa=nothing)

The per-vector 8-bit [`QuantDatabase`](@ref SimilaritySearch.ScalarQuant.QuantDatabase): one `UInt8` code per coordinate, each
stored vector carrying its **own** `min`/scale pair (`E[i]`) computed from that vector's own
extrema. Indexing yields an [`SQu8Vec`](@ref), which the distances consume without
dequantizing.

The first constructor quantizes `X` (it is what [`quantize`](@ref) calls) into the storage
`storage` builds from the matrix of codes. The second takes the two fields back as they are,
quantizing nothing -- that is the one to use after reading `E`/`Q` from storage, and, with
an empty growable `Q`, the one that starts a database `push_item!` can grow: see
[`QuantDatabase`](@ref SimilaritySearch.ScalarQuant.QuantDatabase) for the storage choices and the growth
interface. The two must agree: exactly one `SQMinC` per stored vector.
"""
const SQu8Database = QuantDatabase{8,Vector{SQMinC}}

"""
    quantize(X::AbstractMatrix; storage=MatrixDatabase)

Scalar-quantizes each column (vector) of `X` to 8 bits per coordinate, one `UInt8` code per coordinate.
This reduces the memory footprint of a database of vectors by roughly a factor of
4 with respect to `Float32` at the cost of precision. Each column is quantized
independently using its own minimum and scale factor, computed from the extrema of the
column so that its whole range `[min, max]` maps onto the codes.

Returns an [`SQu8Database`](@ref), an `AbstractDatabase` whose `db[i]` is an [`SQu8Vec`](@ref)
that can be indexed to retrieve dequantized `Float32` coordinates; `storage` chooses the
database that holds the codes (see [`QuantDatabase`](@ref SimilaritySearch.ScalarQuant.QuantDatabase)).

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
quantize(X::AbstractMatrix; kwargs...) = SQu8Database(X; kwargs...)

"""
    quantize(db::SQu8Database, v::AbstractVector)

Quantizes a single vector `v` to 8 bits per coordinate, the same way as the vectors already
stored in `db`, returning an [`SQu8Vec`](@ref). Each vector's own `min`/scale comes from its
own extrema, so this does not read `db`'s parameters; `db` only fixes the (padded)
dimension `v` must have. It is what [`push_item!`](@ref SimilaritySearch.ScalarQuant.QuantDatabase) stores,
and how a query is quantized to be compared as codes against codes with
[`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref)/[`NormCosine`](@ref).

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8.quantize(X);

julia> qv = ScalarQuant.SQu8.quantize(db, rand(Float32, 8));  # quantized the same way as db's vectors
```
"""
quantize(db::SQu8Database, v::AbstractVector) = ScalarQuant.quantize(db, v)

end
