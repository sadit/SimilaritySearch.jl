module Projections

using Polyester, Random, LinearAlgebra, Distributions, StatsBase
export RandomProjections, outdim, indim, transform, transform!
using ...SimilaritySearch.Dist.CastF32: dot32
using ...SimilaritySearch: @BATCH

"""
    RandomProjections(map::M) where {M<:AbstractMatrix}

Wraps a projection matrix `map` of size `(indim, outdim)` used to reduce the
dimension of vectors from `indim` to `outdim` via a linear projection (`v -> map' * v`).
This is a standard dimensionality-reduction technique for similarity search: projecting
onto a lower-dimensional space reduces both memory usage and distance-computation cost,
while approximately preserving relative distances (see the Johnson-Lindenstrauss lemma).

Use [`gaussian`](@ref) or [`qr`](@ref) to build a `RandomProjections` map, and
[`transform`](@ref)/[`transform!`](@ref) to apply it to vectors or matrices of vectors.

# Arguments
- `map`: the projection matrix, with `indim` rows and `outdim` columns

# Examples

```julia
julia> using SimilaritySearch

julia> rp = Special.Projections.gaussian(128, 32);  # random gaussian projection 128 -> 32

julia> rp2 = Special.Projections.qr(128, 32);  # QR-orthogonalized projection 128 -> 32
```
"""
struct RandomProjections{M<:AbstractMatrix}
    map::M
end

getmap(rp::RandomProjections) = rp.map

"""
    gaussian(rng::AbstractRNG, FloatType::Type, indim::Int, outdim::Int)
    gaussian(indim::Int, outdim::Int=indim)

Builds a [`RandomProjections`](@ref) whose map is a dense `indim × outdim` matrix with
entries drawn independently from a Normal distribution with mean zero and standard
deviation `1/outdim`, whose columns are then normalized to unit norm. This is a
Gaussian random projection: unlike [`qr`](@ref), the resulting columns are not
orthogonal to each other, but generating and applying it is cheaper.

# Arguments
- `rng`: the random number generator to use (defaults to `Random.default_rng()`)
- `FloatType`: the floating point type of the projection matrix (defaults to `Float32`)
- `indim`: the dimension of the input vectors
- `outdim`: the dimension of the projected vectors (defaults to `indim`)

# Examples

```julia
julia> using SimilaritySearch

julia> rp = Special.Projections.gaussian(128, 32);

julia> size(Special.Projections.getmap(rp))
(128, 32)
```
"""
function gaussian(rng::AbstractRNG, FloatType::Type, indim::Int, outdim::Int)
    N = Normal(zero(FloatType), convert(FloatType, 1 / outdim))
    M = rand(rng, N, indim, outdim)
    for c in eachcol(M)
        normalize!(c)
    end

    RandomProjections(M)
end

"""
    qr(rng::AbstractRNG, FloatType::Type, indim::Int, outdim::Int)
    qr(indim::Int, outdim::Int=indim)

Builds a [`RandomProjections`](@ref) whose map is the (first `outdim` columns of the)
`Q` factor of the QR decomposition of a random `indim × indim` matrix. Unlike
[`gaussian`](@ref), the resulting projection matrix has orthonormal columns, which makes
the projection an isometry up to the subspace it projects onto (distances between
projected vectors are not shrunk by non-orthogonality), at the extra cost of computing
the QR factorization.

# Arguments
- `rng`: the random number generator to use (defaults to `Random.default_rng()`)
- `FloatType`: the floating point type of the projection matrix (defaults to `Float32`)
- `indim`: the dimension of the input vectors
- `outdim`: the dimension of the projected vectors (defaults to `indim`)
"""
function qr(rng::AbstractRNG, FloatType::Type, indim::Int, outdim::Int)
    M, _ = LinearAlgebra.qr(rand(rng, FloatType, (indim, indim)))
    M = Matrix(M)

    if indim != outdim
        RandomProjections(M[:, 1:outdim])
    else
        RandomProjections(M)
    end
end

gaussian(indim::Int, outdim::Int=indim) = gaussian(Random.default_rng(), Float32, indim, outdim)
qr(indim::Int, outdim::Int=indim) = qr(Random.default_rng(), Float32, indim, outdim)

Base.size(rp::RandomProjections) = size(getmap(rp))

"""
    indim(rp::RandomProjections)

Returns the input dimension of the projection `rp`, i.e., the dimension that vectors
passed to [`transform`](@ref)/[`transform!`](@ref) are expected to have.
"""
indim(rp::RandomProjections) = size(getmap(rp), 1)

"""
    outdim(rp::RandomProjections)

Returns the output dimension of the projection `rp`, i.e., the dimension of the vectors
produced by [`transform`](@ref)/[`transform!`](@ref).
"""
outdim(rp::RandomProjections) = size(getmap(rp), 2)
Base.eltype(rp::RandomProjections) = eltype(getmap(rp))

"""
    transform!(rp::RandomProjections, out::AbstractVector, v::AbstractVector)

In-place version of [`transform`](@ref): projects `v` using `rp` and stores the result
in `out`, which must have length `outdim(rp)`. Returns `out`.

# Arguments
- `rp`: the projection to apply
- `out`: the output vector where the projected vector is stored
- `v`: the input vector to project, of length `indim(rp)`
"""
function transform!(rp::RandomProjections, out::AbstractVector, v::AbstractVector)
    for (i, x) in enumerate(eachcol(getmap(rp)))
        @inbounds out[i] = dot32(x, v)
    end

    out
end

"""
    transform(rp::RandomProjections, v::AbstractVector)

Projects the vector `v` (of length `indim(rp)`) using `rp`, returning a new vector of
length `outdim(rp)`. Each output coordinate is the dot product of `v` with the
corresponding column of the projection map.

# Arguments
- `rp`: the projection to apply
- `v`: the input vector to project
"""
function transform(rp::RandomProjections, v::AbstractVector)
    out = Vector{eltype(rp)}(undef, outdim(rp))
    transform!(rp, out, v)
end

"""
    transform(rp::RandomProjections, X::AbstractMatrix; minbatch::Int=4)

Projects every column (vector) of `X` using `rp`, returning a new matrix with
`outdim(rp)` rows and the same number of columns as `X`. Columns are projected in
parallel using `@BATCH`.

# Arguments
- `rp`: the projection to apply
- `X`: a matrix whose columns are the vectors to project, each of length `indim(rp)`
- `minbatch`: minimum number of columns processed per parallel task (see `@BATCH`)

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 128, 1000);

julia> rp = Special.Projections.gaussian(128, 32);

julia> Y = Special.Projections.transform(rp, X);

julia> size(Y)
(32, 1000)
```
"""
function transform(rp::RandomProjections, X::AbstractMatrix; minbatch::Int=4)
    O = Matrix{eltype(rp)}(undef, outdim(rp), size(X, 2))
    transform!(rp, O, X; minbatch)
end

"""
    transform!(rp::RandomProjections, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)

In-place version of `transform(rp, X)`: projects every column of `X` using `rp` and
stores the result in `O`, which must have `outdim(rp)` rows and the same number of
columns as `X`. Returns `O`.

# Arguments
- `rp`: the projection to apply
- `O`: the output matrix where the projected vectors are stored
- `X`: a matrix whose columns are the vectors to project, each of length `indim(rp)`
- `minbatch`: minimum number of columns processed per parallel task (see `@BATCH`)
"""
function transform!(rp::RandomProjections, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)

    @BATCH minbatch=minbatch for i in 1:n
        o = view(O, :, i)
        x = view(X, :, i)
        transform!(rp, o, x)
    end

    O
end

include("hadamard.jl")
include("bitsketches.jl")

end