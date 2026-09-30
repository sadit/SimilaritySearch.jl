
export HadamardProjection, indim, outdim, transform, transform!

using SIMD: Vec, vload, vstore, shufflevector, vifelse

"""
    fwht!(a::AbstractVector) -> a

The fast Walsh-Hadamard transform of `a` in place, in natural (Hadamard) ordering and
scaled by `1/length(a)`: the same values `Hadamard.jl`'s `fwht_natural!` produces, bit for
bit, which is what every encoding built on [`HadamardProjection`](@ref) was measured with.
`length(a)` must be a power of two (unchecked here; the projection's constructor checks).

It is the plain iterative butterfly, `log2(n)` passes of sums and differences over halves of
growing size, and it needs no plan. That is the point of it: the transform used to go
through FFTW, which built a plan under its global planning lock on **every** per-vector
call (170-400 µs per vector at 128-4096 dimensions, and worse with threads, which serialize
on the lock), and whose execution of this shape -- a 2×2×…×2 multidimensional `r2r`
transform -- stays at 3-54 µs even with a cached, `MEASURE`-quality plan (issue #89). Don't
bring a planned transform back for this.

For a contiguous `Float32`/`Float64` vector (a `Vector`, or a column view of a matrix) the
first three passes, whose inner loops have 1, 2 and 4 elements and cannot vectorize, are one
sweep of 8-point blocks done with three lane shuffles of a `Vec{8}`, the remaining passes
load and store `Vec{8}`s, and the `1/n` scale rides in the last pass: 95 ns at 128
dimensions, 0.4 µs at 512, 1.0 µs at 1024 and 4.6 µs at 4096 in `Float32` (33-800 ns per
column of a matrix over 64 threads), 4-5x the scalar loops, which any other element type or
layout still gets. Every variant is bit-identical:
each output is the same sum or difference of the same two values, in the same order.
"""
fwht!(a::AbstractVector) = _scale!(_passes!(a, 1))

const _SIMDFloat = Union{Float32,Float64}
const _ContiguousVector{T} = Union{DenseVector{T},Base.FastContiguousSubArray{T,1}}

function fwht!(a::_ContiguousVector{T}) where {T<:_SIMDFloat}
    n = length(a)
    if n >= 16
        _wht8blocks!(a)
        _passes_vec!(a, 8)
    elseif n == 8
        _scale!(_wht8blocks!(a))
    else
        _scale!(_passes!(a, 1))
    end
    a
end

"The scalar passes from `h = h0` upward."
@inline function _passes!(a::AbstractVector{T}, h0::Int) where {T}
    n = length(a)
    h = h0
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
    a
end

@inline function _scale!(a::AbstractVector{T}) where {T}
    s = one(T) / length(a)
    @inbounds @simd for i in eachindex(a)
        a[i] *= s
    end
    a
end

# the passes h = 1, 2, 4 over one 8-point block held in a register: at each level the partner
# of lane j is lane j ⊻ h, brought in by a shuffle, and the lanes whose bit h is clear take
# the sum while the others take the difference
const _WHT8_M1 = Vec{8,Bool}((true, false, true, false, true, false, true, false))
const _WHT8_M2 = Vec{8,Bool}((true, true, false, false, true, true, false, false))
const _WHT8_M4 = Vec{8,Bool}((true, true, true, true, false, false, false, false))

@inline function _wht8(v::Vec{8,T}) where {T}
    w = shufflevector(v, Val((1, 0, 3, 2, 5, 4, 7, 6)))
    v = vifelse(_WHT8_M1, v + w, w - v)
    w = shufflevector(v, Val((2, 3, 0, 1, 6, 7, 4, 5)))
    v = vifelse(_WHT8_M2, v + w, w - v)
    w = shufflevector(v, Val((4, 5, 6, 7, 0, 1, 2, 3)))
    vifelse(_WHT8_M4, v + w, w - v)
end

"The first three passes as one sweep of 8-point blocks; `length(a)` a multiple of 8."
@inline function _wht8blocks!(a::_ContiguousVector{T}) where {T}
    @inbounds for i in 1:8:length(a)
        vstore(_wht8(vload(Vec{8,T}, a, i)), a, i)
    end
    a
end

"The passes from `h = h0 >= 8` upward on `Vec{8}`s, the `1/n` scale applied in the last one."
@inline function _passes_vec!(a::_ContiguousVector{T}, h0::Int) where {T}
    n = length(a)
    h = h0
    @inbounds while h < n
        last = 2h >= n
        s = last ? one(T) / n : one(T)
        i = 1
        while i <= n
            j = i
            while j < i + h
                x = vload(Vec{8,T}, a, j)
                y = vload(Vec{8,T}, a, j + h)
                if last
                    vstore((x + y) * s, a, j)
                    vstore((x - y) * s, a, j + h)
                else
                    vstore(x + y, a, j)
                    vstore(x - y, a, j + h)
                end
                j += 8
            end
            i += 2h
        end
        h *= 2
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
