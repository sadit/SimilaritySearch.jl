"""
    SQu4

Per-vector (per-column) 4-bit scalar quantization: [`quantize`](@ref SQu4.quantize) stores
two 4-bit codes packed per `UInt8`, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu4.quantize`, etc. 

The vector type, the database and the distances are the module-wide
[`SQVec`](@ref ScalarQuant.SQVec), [`QuantDatabase`](@ref ScalarQuant.QuantDatabase) and
[`SqL2`](@ref ScalarQuant.SqL2)/[`L2`](@ref ScalarQuant.L2)/[`L1`](@ref ScalarQuant.L1)/
[`NormCosine`](@ref ScalarQuant.NormCosine); the names here are aliases kept for the
per-width API.
"""
module SQu4

export quantize, SQu4Vec, SQu4Database, L1, L2, SqL2, NormCosine

import ..ScalarQuant
using ..ScalarQuant: SQMinC, SQVec, QuantDatabase, L1, L2, SqL2, NormCosine

"4-bit [`SQVec`](@ref ScalarQuant.SQVec), two 4-bit codes packed per `UInt8`. `SQu4Vec(v)` quantizes `v` on its own extrema."
const SQu4Vec = SQVec{4}

"""
    SQu4Database(X::AbstractMatrix; storage=MatrixDatabase)
    SQu4Database(E::AbstractVector{SQMinC}, Q; dim=nothing, Sa=nothing, Saa=nothing)

The per-vector 4-bit [`QuantDatabase`](@ref ScalarQuant.QuantDatabase): two 4-bit codes packed per `UInt8`, each
stored vector carrying its **own** `min`/scale pair (`E[i]`) computed from that vector's own
extrema. Indexing yields an [`SQu4Vec`](@ref), which the distances consume without
dequantizing.

The first constructor quantizes `X` (it is what [`quantize`](@ref) calls) into the storage
`storage` builds from the matrix of codes. The second takes the two fields back as they are,
quantizing nothing -- that is the one to use after reading `E`/`Q` from storage, and, with
an empty growable `Q`, the one that starts a database `push_item!` can grow: see
[`QuantDatabase`](@ref ScalarQuant.QuantDatabase) for the storage choices and the growth
interface. The two must agree: exactly one `SQMinC` per stored vector.
"""
const SQu4Database = QuantDatabase{4,Vector{SQMinC}}

"""
    quantize(X::AbstractMatrix; storage=MatrixDatabase)

Scalar-quantizes each column (vector) of `X` to 4 bits per coordinate, two 4-bit codes packed per `UInt8`.
This reduces the memory footprint of a database of vectors by roughly a factor of
8 with respect to `Float32` at the cost of precision. Each column is quantized
independently using its own minimum and scale factor, computed from the extrema of the
column so that its whole range `[min, max]` maps onto the codes; `size(X, 1)` must be a multiple of 2.

Returns an [`SQu4Database`](@ref), an `AbstractDatabase` whose `db[i]` is an [`SQu4Vec`](@ref)
that can be indexed to retrieve dequantized `Float32` coordinates; `storage` chooses the
database that holds the codes (see [`QuantDatabase`](@ref ScalarQuant.QuantDatabase)).

!!! note
    If `X` needs padding to a multiple of 2, any plain (non-quantized) query vector later
    compared against the database via the mixed distances must be padded to that same
    dimension too, since they index the plain vector positionally and do not know about
    the padding.

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu4.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
quantize(X::AbstractMatrix; kwargs...) = SQu4Database(X; kwargs...)

"""
    quantize(db::SQu4Database, v::AbstractVector)

Quantizes a single vector `v` to 4 bits per coordinate, the same way as the vectors already
stored in `db`, returning an [`SQu4Vec`](@ref). Each vector's own `min`/scale comes from its
own extrema, so this does not read `db`'s parameters; `db` only fixes the (padded)
dimension `v` must have. It is what [`push_item!`](@ref ScalarQuant.QuantDatabase) stores,
and how a query is quantized to be compared as codes against codes with
[`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref)/[`NormCosine`](@ref).

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu4.quantize(X);

julia> qv = ScalarQuant.SQu4.quantize(db, rand(Float32, 8));  # quantized the same way as db's vectors
```
"""
quantize(db::SQu4Database, v::AbstractVector) = ScalarQuant.quantize(db, v)

end
