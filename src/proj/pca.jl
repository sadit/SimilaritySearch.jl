# This file is a part of SimilaritySearch.jl

using MultivariateStats: MultivariateStats

export PCAProjection

"""
    PCAProjection(X::AbstractMatrix{<:AbstractFloat}, outdim::Int; kwargs...)

Wraps a PCA-based dimensionality reduction (via MultivariateStats.jl's `PCA`), analogous
in purpose to [`RandomProjections`](@ref) but fitted from data instead of drawn at random:
[`transform`](@ref)/[`transform!`](@ref) projects a vector/matrix onto the `outdim`
orthogonal directions of largest variance in `X` (after centering by `X`'s mean). Unlike a
random projection, the reduction this produces depends on, and is tailored to, the
specific data it was fitted on -- so, unlike [`RandomProjections`](@ref)/
[`HadamardProjection`](@ref), it cannot be regenerated independently of that data; reuse
the same `PCAProjection` (as with those, e.g. via [`transform`](@ref)) to project a query
comparably to an already-projected dataset.

# Arguments
- `X`: the training data, one object per column, used to fit the principal directions
- `outdim`: the number of output dimensions to keep (MultivariateStats' `maxoutdim`)
- `kwargs...`: forwarded to `MultivariateStats.fit(MultivariateStats.PCA, X;
  maxoutdim=outdim, kwargs...)` (e.g. `method=:svd`/`:cov`, `pratio`, `mean`)

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 128, 1000);

julia> p = SimilaritySearch.Projections.PCAProjection(X, 32);

julia> indim(p), outdim(p)
(128, 32)

julia> Y = SimilaritySearch.Projections.transform(p, X);

julia> size(Y)
(32, 1000)
```
"""
struct PCAProjection{M<:MultivariateStats.PCA}
    pca::M
end

function PCAProjection(X::AbstractMatrix{<:AbstractFloat}, outdim::Int; kwargs...)
    PCAProjection(MultivariateStats.fit(MultivariateStats.PCA, X; maxoutdim=outdim, kwargs...))
end

getmap(p::PCAProjection) = p.pca
Base.size(p::PCAProjection) = (indim(p), outdim(p))
Base.eltype(p::PCAProjection) = eltype(MultivariateStats.projection(p.pca))

"""
    indim(p::PCAProjection)

Returns the input dimension of the projection `p`, i.e., the dimension that vectors
passed to [`transform`](@ref)/[`transform!`](@ref) are expected to have.
"""
indim(p::PCAProjection) = size(p.pca, 1)

"""
    outdim(p::PCAProjection)

Returns the output dimension of the projection `p`, i.e., the dimension of the vectors
produced by [`transform`](@ref)/[`transform!`](@ref).
"""
outdim(p::PCAProjection) = size(p.pca, 2)

"""
    transform(p::PCAProjection, v::AbstractVector)

Projects the vector `v` (of length `indim(p)`) onto `p`'s principal directions, returning
a new vector of length `outdim(p)`.
"""
transform(p::PCAProjection, v::AbstractVector) = MultivariateStats.predict(p.pca, v)

"""
    transform(p::PCAProjection, X::AbstractMatrix)

Projects every column (vector) of `X` onto `p`'s principal directions, returning a new
matrix with `outdim(p)` rows and the same number of columns as `X`. Unlike
[`RandomProjections`](@ref)/[`HadamardProjection`](@ref), this is a single call into
MultivariateStats (already a vectorized BLAS matrix-matrix product), so there is no
`minbatch` to parallelize over.

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 128, 1000);

julia> p = SimilaritySearch.Projections.PCAProjection(X, 32);

julia> Y = SimilaritySearch.Projections.transform(p, X);

julia> size(Y)
(32, 1000)
```
"""
transform(p::PCAProjection, X::AbstractMatrix) = MultivariateStats.predict(p.pca, X)

"""
    transform!(p::PCAProjection, out::AbstractVector, v::AbstractVector)
    transform!(p::PCAProjection, O::AbstractMatrix, X::AbstractMatrix)

In-place versions of [`transform`](@ref): projects `v`/`X` using `p` and stores the result
in `out`/`O`, which must have length/row-count `outdim(p)`. Returns `out`/`O`.
"""
transform!(p::PCAProjection, out::AbstractVector, v::AbstractVector) = copyto!(out, transform(p, v))
transform!(p::PCAProjection, O::AbstractMatrix, X::AbstractMatrix) = copyto!(O, transform(p, X))
