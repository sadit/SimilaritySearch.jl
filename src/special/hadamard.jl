module HadamardProjection

using Polyester
using Hadamard: fwht

export Projection, indim, outdim, transform, transform!

"""
    Projection(indim::Int, outdim::Int=indim)

Wraps a fast Walsh-Hadamard transform (FWHT), used as a dimensionality-reduction
projection from `indim` to `outdim` (via [`transform`](@ref)/[`transform!`](@ref)),
analogous in purpose to [`Special.Projections.RandomProjections`](@ref) but computed
with the ``O(n \\log n)`` FWHT (via `Hadamard.fwht`) instead of a dense matrix-vector
product, and requiring no random matrix to be generated or stored.

`transform` applies the full-size FWHT to the input vector (in the sequency ordering,
i.e., ordered by number of sign changes, roughly analogous to increasing frequency in a
Fourier transform) and keeps only the first `outdim` coefficients, which concentrate most
of the energy of typical (smooth/correlated) inputs. When `outdim == indim` (the
default), the transform is a full, exact (up to normalization), orthogonal change of
basis and no information is discarded.

# Arguments
- `indim`: the dimension of the input vectors; must be a power of two (`fwht`
  requirement), otherwise an `ArgumentError` is thrown
- `outdim`: the dimension of the projected vectors (defaults to `indim`); must satisfy
  `0 < outdim <= indim`, otherwise an `ArgumentError` is thrown

# Examples

```julia
julia> using SimilaritySearch

julia> hp = Special.HadamardProjection.Projection(128, 32);  # 128 -> 32

julia> Special.HadamardProjection.indim(hp), Special.HadamardProjection.outdim(hp)
(128, 32)
```
"""
struct Projection
    indim::Int
    outdim::Int

    function Projection(indim::Int, outdim::Int=indim)
        ispow2(indim) || throw(ArgumentError("HadamardProjection.Projection: indim=$indim must be a power of two (the fast Walsh-Hadamard transform only supports power-of-two lengths)"))
        0 < outdim <= indim || throw(ArgumentError("HadamardProjection.Projection: outdim=$outdim must satisfy 0 < outdim <= indim=$indim"))
        new(indim, outdim)
    end
end

Base.size(hp::Projection) = (hp.indim, hp.outdim)

"""
    indim(hp::Projection)

Returns the input dimension of the projection `hp`, i.e., the dimension that vectors
passed to [`transform`](@ref)/[`transform!`](@ref) are expected to have.
"""
indim(hp::Projection) = hp.indim

"""
    outdim(hp::Projection)

Returns the output dimension of the projection `hp`, i.e., the dimension of the vectors
produced by [`transform`](@ref)/[`transform!`](@ref).
"""
outdim(hp::Projection) = hp.outdim

"""
    transform!(hp::Projection, out::AbstractVector, v::AbstractVector)

In-place version of [`transform`](@ref): projects `v` using `hp` and stores the result
in `out`, which must have length `outdim(hp)`. Returns `out`.

# Arguments
- `hp`: the projection to apply
- `out`: the output vector where the projected vector is stored, of length `outdim(hp)`
- `v`: the input vector to project, of length `indim(hp)`
"""
function transform!(hp::Projection, out::AbstractVector, v::AbstractVector)
    length(v) == indim(hp) || throw(DimensionMismatch("HadamardProjection.transform!: length(v)=$(length(v)) must equal indim(hp)=$(indim(hp))"))
    length(out) == outdim(hp) || throw(DimensionMismatch("HadamardProjection.transform!: length(out)=$(length(out)) must equal outdim(hp)=$(outdim(hp))"))

    y = fwht(v)
    if outdim(hp) == indim(hp)
        copyto!(out, y)
    else
        copyto!(out, view(y, 1:outdim(hp)))
    end

    out
end

"""
    transform(hp::Projection, v::AbstractVector)

Projects the vector `v` (of length `indim(hp)`) using `hp`, returning a new vector of
length `outdim(hp)`. Computed as the fast Walsh-Hadamard transform of `v`, truncated to
its first `outdim(hp)` (sequency-ordered) coefficients.

# Arguments
- `hp`: the projection to apply
- `v`: the input vector to project
"""
function transform(hp::Projection, v::AbstractVector)
    out = Vector{float(eltype(v))}(undef, outdim(hp))
    transform!(hp, out, v)
end

"""
    transform(hp::Projection, X::AbstractMatrix; minbatch::Int=4)

Projects every column (vector) of `X` using `hp`, returning a new matrix with
`outdim(hp)` rows and the same number of columns as `X`. Columns are projected in
parallel using `Polyester.@batch`.

# Arguments
- `hp`: the projection to apply
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: minimum number of columns processed per parallel task (see `Polyester.@batch`)

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 128, 1000);

julia> hp = Special.HadamardProjection.Projection(128, 32);

julia> Y = Special.HadamardProjection.transform(hp, X);

julia> size(Y)
(32, 1000)
```
"""
function transform(hp::Projection, X::AbstractMatrix; minbatch::Int=4)
    O = Matrix{float(eltype(X))}(undef, outdim(hp), size(X, 2))
    transform!(hp, O, X; minbatch)
end

"""
    transform!(hp::Projection, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)

In-place version of `transform(hp, X)`: projects every column of `X` using `hp` and
stores the result in `O`, which must have `outdim(hp)` rows and the same number of
columns as `X`. Returns `O`.

# Arguments
- `hp`: the projection to apply
- `O`: the output matrix where the projected vectors are stored
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: minimum number of columns processed per parallel task (see `Polyester.@batch`)
"""
function transform!(hp::Projection, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)

    @batch per = thread minbatch = minbatch for i in 1:n
        o = view(O, :, i)
        x = view(X, :, i)
        transform!(hp, o, x)
    end

    O
end

end
