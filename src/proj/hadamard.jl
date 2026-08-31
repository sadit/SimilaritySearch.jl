
using Hadamard: fwht_natural!

export HadamardProjection, indim, outdim, transform, transform!

"""
    HadamardProjection(indim::Int)
    HadamardProjection(indim::Int, outdim::Int)

Wraps a fast Walsh-Hadamard transform (FWHT), used as an orthogonal change of basis (via
[`transform`](@ref)/[`transform!`](@ref)), analogous in purpose to
[`RandomProjections`](@ref) but computed with the ``O(n \\log n)`` FWHT (via
`Hadamard.fwht_natural!`) instead of a dense matrix-vector product, and requiring no random
matrix to be generated or stored.

Unlike [`RandomProjections`](@ref), `HadamardProjection` does **not** reduce
dimensionality: `transform` always returns as many coordinates as it received
(`outdim(hp) == indim(hp)`), since `fwht_natural!` computes a full, exact (up to normalization),
orthogonal transform of its input, in natural Hadamard ordering. The
two-argument constructor exists only to make `outdim` explicit at call sites that already
pass one to other projection types (e.g. [`RandomProjections`](@ref)); it requires
`outdim == indim` and raises `ArgumentError` otherwise.

# Arguments
- `indim`: the dimension of the input vectors; must be a power of two (`fwht_natural!`
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

    if out !== v
        copyto!(out, v)
    end
    fwht_natural!(out)
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
size as `X`. Computed as a single Walsh-Hadamard transform batched over every column
at once (see [`transform!`](@ref)), not as a per-column loop.

# Arguments
- `hp`: the projection to apply
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: accepted for interface symmetry with other `transform` methods, but unused
  (see [`transform!`](@ref))

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

Applies a single Walsh-Hadamard transform to the whole matrix at once (`Hadamard`'s FFTW
plan is built with the `n` columns as a "howmany" batch dimension, along the lines of
[`Hadamard.fwht_natural!`](https://github.com/stevengj/Hadamard.jl)'s `region` argument),
rather than looping `n` times over one column each -- looping would rebuild an FFTW plan
under FFTW's global planning lock on every single column, which is both unamortized (paying
full plan-construction overhead per vector instead of once for the batch) and unparallelizable
(every thread serializes on that lock); see issue #54.

# Arguments
- `hp`: the projection to apply
- `O`: the output matrix where the projected vectors are stored
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: accepted for interface symmetry with other `transform!` methods (e.g.
  [`RandomProjections`](@ref)), but unused: a single batched FWHT call already processes
  every column without needing to be split across tasks
"""
function transform!(hp::HadamardProjection, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    size(X, 1) == indim(hp) || throw(DimensionMismatch("HadamardProjection.transform!: size(X,1)=$(size(X,1)) must equal indim(hp)=$(indim(hp))"))
    size(O) == size(X) || throw(DimensionMismatch("HadamardProjection.transform!: size(O)=$(size(O)) must equal size(X)=$(size(X))"))

    if O !== X
        copyto!(O, X)
    end
    fwht_natural!(O, (1,))
    O
end
