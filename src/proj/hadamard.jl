
export HadamardProjection, indim, outdim, transform, transform!

"""
    fwht!(a::AbstractVector) -> a

The fast Walsh-Hadamard transform of `a` in place, in natural (Hadamard) ordering and
scaled by `1/length(a)`: the same values `Hadamard.jl`'s `fwht_natural!` produces, bit for
bit, which is what every encoding built on [`HadamardProjection`](@ref) was measured with.
`length(a)` must be a power of two (unchecked here; the projection's constructor checks).

It is the plain iterative butterfly, `log2(n)` passes of sums and differences over halves of
growing size, which LLVM vectorizes, and it needs no plan. That is the point of it: the
transform used to go through FFTW, which built a plan under its global planning lock on
**every** per-vector call (170-400 µs per vector at 128-4096 dimensions, and worse with
threads, which serialize on the lock), and whose execution of this shape -- a 2×2×…×2
multidimensional `r2r` transform -- stays at 3-54 µs even with a cached, `MEASURE`-quality
plan. The butterfly is 0.5-15 µs per vector on one thread and 30-230 ns per column over 64
threads, 50x the batched FFTW call the matrix path used to make (issue #89). Don't bring a
planned transform back for this.
"""
function fwht!(a::AbstractVector{T}) where {T}
    n = length(a)
    h = 1
    @inbounds while h < n
        i = 1
        while i <= n
            @simd for j in i:i+h-1
                x = a[j]
                y = a[j+h]
                a[j] = x + y
                a[j+h] = x - y
            end
            i += 2h
        end
        h *= 2
    end
    s = one(T) / n
    @inbounds @simd for i in eachindex(a)
        a[i] *= s
    end
    a
end

"""
    HadamardProjection(indim::Int)
    HadamardProjection(indim::Int, outdim::Int)

Wraps a fast Walsh-Hadamard transform (FWHT), used as an orthogonal change of basis (via
[`transform`](@ref)/[`transform!`](@ref)), analogous in purpose to
[`RandomProjections`](@ref) but computed with the ``O(n \\log n)`` FWHT (an in-place
butterfly, [`fwht!`](@ref); no plan, no FFTW) instead of a dense matrix-vector product, and
requiring no random matrix to be generated or stored.

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
    fwht!(out)
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
size as `X`: one [`fwht!`](@ref) per column, in parallel (see [`transform!`](@ref)).

# Arguments
- `hp`: the projection to apply
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: the `@BATCHES` batch size over the columns

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

One [`fwht!`](@ref) per column, over `@BATCHES` of `minbatch` columns, the copy from `X`
included: nothing is planned and nothing is locked, so the columns transform in parallel
(a serial `copyto!(O, X)` ahead of the loop cost three times the transform itself). (Issue #54 had made this path a
single batched FFTW call to avoid rebuilding a plan per column; issue #89 removed the plans
altogether, and the per-column butterfly is 50x faster than that batched call.)

# Arguments
- `hp`: the projection to apply
- `O`: the output matrix where the projected vectors are stored
- `X`: a matrix whose columns are the vectors to project, each of length `indim(hp)`
- `minbatch`: the `@BATCHES` batch size over the columns
"""
function transform!(hp::HadamardProjection, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    size(X, 1) == indim(hp) || throw(DimensionMismatch("HadamardProjection.transform!: size(X,1)=$(size(X,1)) must equal indim(hp)=$(indim(hp))"))
    size(O) == size(X) || throw(DimensionMismatch("HadamardProjection.transform!: size(O)=$(size(O)) must equal size(X)=$(size(X))"))

    if O === X
        @BATCHES minbatch for j in axes(O, 2)
            fwht!(view(O, :, j))
        end
    else
        @BATCHES minbatch for j in axes(O, 2)          # the copy inside the batch too: serial, it cost 3x the transform
            copyto!(view(O, :, j), view(X, :, j))
            fwht!(view(O, :, j))
        end
    end
    O
end
