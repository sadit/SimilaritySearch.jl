
using Hadamard: fwht

export HadamardProjection, indim, outdim, transform, transform!

"""
    HadamardProjection(indim::Int)
    HadamardProjection(indim::Int, outdim::Int)

Wraps a fast Walsh-Hadamard transform (FWHT), used as an orthogonal change of basis (via
[`transform`](@ref)/[`transform!`](@ref)), analogous in purpose to
[`RandomProjections`](@ref) but computed with the ``O(n \\log n)`` FWHT (via
`Hadamard.fwht`) instead of a dense matrix-vector product, and requiring no random
matrix to be generated or stored.

Unlike [`RandomProjections`](@ref), `HadamardProjection` does **not** reduce
dimensionality: `transform` always returns as many coordinates as it received
(`outdim(hp) == indim(hp)`), since `fwht` computes a full, exact (up to normalization),
orthogonal transform of its input, in the sequency ordering (i.e., ordered by number of
sign changes, roughly analogous to increasing frequency in a Fourier transform). The
two-argument constructor exists only to make `outdim` explicit at call sites that already
pass one to other projection types (e.g. [`RandomProjections`](@ref)); it requires
`outdim == indim` and raises `ArgumentError` otherwise.

# Arguments
- `indim`: the dimension of the input vectors; must be a power of two (`fwht`
  requirement), otherwise an `ArgumentError` is thrown
- `outdim`: if given, must equal `indim` (otherwise an `ArgumentError` is thrown), since
  this projection does not support dimensionality reduction/truncation

# Examples

```julia
julia> using SimilaritySearch

julia> hp = Projections.HadamardProjection(128);

julia> Projections.indim(hp), Projections.outdim(hp)
(128, 128)
```
"""
struct HadamardProjection
    indim::Int

    function HadamardProjection(indim::Int)
        ispow2(indim) || throw(ArgumentError("HadamardProjection: indim=$indim must be a power of two (the fast Walsh-Hadamard transform only supports power-of-two lengths)"))
        new(indim)
    end
end

function HadamardProjection(indim::Int, outdim::Int)
    outdim == indim || throw(ArgumentError("HadamardProjection: outdim=$outdim must equal indim=$indim (this projection does not support dimensionality reduction/truncation)"))
    HadamardProjection(indim)
end

Base.size(hp::HadamardProjection) = (hp.indim, hp.indim)

"""
    indim(hp::HadamardProjection)

Returns the input dimension of the projection `hp`, i.e., the dimension that vectors
passed to [`transform`](@ref)/[`transform!`](@ref) are expected to have.
"""
indim(hp::HadamardProjection) = hp.indim

"""
    outdim(hp::HadamardProjection)

Returns the output dimension of the projection `hp`, i.e., the dimension of the vectors
produced by [`transform`](@ref)/[`transform!`](@ref). Always equal to `indim(hp)`, since
`HadamardProjection` does not reduce dimensionality.
"""
outdim(hp::HadamardProjection) = hp.indim

"""
    transform!(hp::HadamardProjection, out::AbstractVector, v::AbstractVector)

In-place version of [`transform`](@ref): projects `v` using `hp` and stores the result
in `out`, which must have length `indim(hp)` (== `outdim(hp)`). Returns `out`.

# Arguments
- `hp`: the projection to apply
- `out`: the output vector where the projected vector is stored, of length `indim(hp)`
- `v`: the input vector to project, of length `indim(hp)`
"""
function transform!(hp::HadamardProjection, out::AbstractVector, v::AbstractVector)
    length(v) == indim(hp) || throw(DimensionMismatch("HadamardProjection.transform!: length(v)=$(length(v)) must equal indim(hp)=$(indim(hp))"))
    length(out) == indim(hp) || throw(DimensionMismatch("HadamardProjection.transform!: length(out)=$(length(out)) must equal indim(hp)=$(indim(hp))"))

    copyto!(out, fwht(v))
end

"""
    transform(hp::HadamardProjection, v::AbstractVector)

Projects the vector `v` (of length `indim(hp)`) using `hp`, returning a new vector of
the same length. Computed as the fast Walsh-Hadamard transform of `v` (sequency-ordered).

# Arguments
- `hp`: the projection to apply
- `v`: the input vector to project
"""
function transform(hp::HadamardProjection, v::AbstractVector)
    out = Vector{float(eltype(v))}(undef, indim(hp))
    transform!(hp, out, v)
end

"""
    transform(hp::HadamardProjection, X::AbstractMatrix; minbatch::Int=4)

Projects every column (vector) of `X` using `hp`, returning a new matrix of the same
size as `X`. Columns are projected in parallel using `@BATCH`.

# Arguments
- `hp`: the projection to apply
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: minimum number of columns processed per parallel task (see `@BATCH`)

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 128, 1000);

julia> hp = Projections.HadamardProjection(128);

julia> Y = Projections.transform(hp, X);

julia> size(Y)
(128, 1000)
```
"""
function transform(hp::HadamardProjection, X::AbstractMatrix; minbatch::Int=4)
    O = Matrix{float(eltype(X))}(undef, indim(hp), size(X, 2))
    transform!(hp, O, X; minbatch)
end

"""
    transform!(hp::HadamardProjection, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)

In-place version of `transform(hp, X)`: projects every column of `X` using `hp` and
stores the result in `O`, which must have the same size as `X`. Returns `O`.

# Arguments
- `hp`: the projection to apply
- `O`: the output matrix where the projected vectors are stored
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: minimum number of columns processed per parallel task (see `@BATCH`)
"""
function transform!(hp::HadamardProjection, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)

    @BATCH minbatch=minbatch for i in 1:n
        o = view(O, :, i)
        x = view(X, :, i)
        transform!(hp, o, x)
    end

    O
end
