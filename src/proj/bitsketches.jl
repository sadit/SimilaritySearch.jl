export bitsketch

"Sets bit `i` (1-indexed) of the `UInt64`-packed bit vector `B`."
@inline function setbit!(B::AbstractVector{UInt64}, i::Integer)
    w = ((i - 1) >> 6) + 1
    b = (i - 1) & 63
    @inbounds B[w] |= (one(UInt64) << b)
    B
end

"""
    packsigns!(out::AbstractVector{UInt64}, y::AbstractVector{<:Real})

Packs the sign of each entry of `y` into `out` (a pre-allocated, `cld(length(y), 64)`-long
`UInt64` vector): entries `>= 0` are encoded as bit `1`, negative entries as bit `0`. If
`length(y)` is not a multiple of 64, the unused high bits of the last word are left as
`0`. Returns `out`.
"""
function packsigns!(out::AbstractVector{UInt64}, y::AbstractVector{<:Real})
    fill!(out, zero(UInt64))
    @inbounds for i in eachindex(y)
        y[i] >= zero(eltype(y)) && setbit!(out, i)
    end

    out
end

"""
    packsigns(y::AbstractVector{<:Real}) -> Vector{UInt64}

Out-of-place version of [`packsigns!`](@ref): packs the sign of each entry of `y` into a
freshly allocated `Vector{UInt64}` of length `cld(length(y), 64)`.
"""
packsigns(y::AbstractVector{<:Real}) = packsigns!(zeros(UInt64, cld(length(y), 64)), y)

"""
    packsigns(Y::AbstractMatrix{<:Real}; minbatch::Int=4) -> Matrix{UInt64}

Packs the sign of every column of `Y` (see [`packsigns`](@ref)) into a `Matrix{UInt64}`
with `cld(size(Y, 1), 64)` rows and the same number of columns as `Y`. Columns are packed
in parallel using `@BATCH`.
"""
function packsigns(Y::AbstractMatrix{<:Real}; minbatch::Int=4)
    m, n = size(Y)
    B = Matrix{UInt64}(undef, cld(m, 64), n)

    @BATCH minbatch=minbatch for j in 1:n
        packsigns!(view(B, :, j), view(Y, :, j))
    end

    B
end

# shared implementation: sketch = sign bits of `transform(rp, data)`, for anything `rp`
# that `transform` accepts (`RandomProjections`, `HadamardProjection`)
_bitsketch_apply(rp, v::AbstractVector) = packsigns(transform(rp, v))
_bitsketch_apply(rp, X::AbstractMatrix; minbatch::Int=4) = packsigns(transform(rp, X; minbatch); minbatch)

"""
    bitsketch(R::AbstractMatrix{<:AbstractFloat}, v::AbstractVector{<:AbstractFloat}) -> Vector{UInt64}
    bitsketch(R::AbstractMatrix{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}; minbatch::Int=4) -> Matrix{UInt64}

Computes a random-rotation bit sketch (a SimHash-style binary locality-sensitive hash):
rotates the input by `R` (the same convention as [`RandomProjections`](@ref), i.e.
`v -> R' * v`) and encodes the *sign* of each of the `size(R, 2)` resulting coordinates
as one bit -- non-negative maps to `1`, negative maps to `0` (see [`packsigns`](@ref)) --
packed into `UInt64` words (64 bits per word, `cld(size(R, 2), 64)` words per sketch).
Vectors whose rotated coordinates fall on the same side of the random hyperplanes defined
by the columns of `R` hash to the same bit pattern, so the Hamming distance between two
sketches approximates the angular distance between the corresponding original vectors.

# Arguments
- `R`: the rotation matrix, of size `(indim, outdim)`; build one with [`gaussian`](@ref)
  or [`qr`](@ref) (pass its `.map`), or use [`bitsketch(method, outdim, data)`](@ref) to
  build and apply a fresh one in a single step
- `v`/`X`: the vector, or matrix (one vector per column), to sketch; must be given as
  `Float32`/`Float64` (or another `AbstractFloat` subtype)
- `minbatch`: (matrix method only) minimum number of columns processed per parallel task

!!! note
    To produce sketches that are meaningfully comparable via Hamming distance (e.g. a
    query sketch against an already-sketched dataset), `R` must be the exact same matrix
    used to sketch that dataset -- reuse it (or the [`RandomProjections`](@ref) object
    wrapping it) rather than generating a new one.

# Examples

```julia
julia> using SimilaritySearch

julia> R = Projections.gaussian(128, 256).map;

julia> X = rand(Float32, 128, 1000);

julia> B = Projections.bitsketch(R, X);

julia> size(B), eltype(B)  # (4, 1000), UInt64  -- cld(256, 64) == 4 words per sketch
```
"""
bitsketch(R::AbstractMatrix{<:AbstractFloat}, v::AbstractVector{<:AbstractFloat}) =
    _bitsketch_apply(RandomProjections(R), v)
bitsketch(R::AbstractMatrix{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}; minbatch::Int=4) =
    _bitsketch_apply(RandomProjections(R), X; minbatch)

"""
    bitsketch(rp::RandomProjections, v::AbstractVector{<:AbstractFloat}) -> Vector{UInt64}
    bitsketch(rp::RandomProjections, X::AbstractMatrix{<:AbstractFloat}; minbatch::Int=4) -> Matrix{UInt64}

Computes a bit sketch (see [`bitsketch(R::AbstractMatrix, data)`](@ref)) using an
already-built [`RandomProjections`](@ref) rotation `rp`, equivalent to
`bitsketch(getmap(rp), data)`. Reuse the same `rp` to sketch a query the same way as an
already-sketched dataset, so that the resulting sketches are comparable.

# Examples

```julia
julia> using SimilaritySearch

julia> rp = Projections.gaussian(128, 256);

julia> X = rand(Float32, 128, 1000);

julia> B = Projections.bitsketch(rp, X);

julia> q = rand(Float32, 128);

julia> bq = Projections.bitsketch(rp, q);  # same rotation, comparable to B's columns
```
"""
bitsketch(rp::RandomProjections, v::AbstractVector{<:AbstractFloat}) = _bitsketch_apply(rp, v)
bitsketch(rp::RandomProjections, X::AbstractMatrix{<:AbstractFloat}; minbatch::Int=4) = _bitsketch_apply(rp, X; minbatch)

"""
    bitsketch(method::Symbol, outdim::Int, data::AbstractVecOrMat{<:AbstractFloat};
              rng::AbstractRNG=Random.default_rng(), FloatType::Type=Float32, minbatch::Int=4)
        -> (bitsketch, R)

Convenience [`bitsketch`](@ref) that builds a fresh rotation matrix `R` -- using
[`gaussian`](@ref) (`method = :gaussian`) or [`qr`](@ref) (`method = :qr`) -- applies it
in a single step, and returns **both** the resulting sketch(es) and `R` as a tuple
`(bitsketch, R)`, so that the very same rotation can be reused afterwards (e.g. via
[`bitsketch(R, data)`](@ref)) to sketch further vectors comparable to this one (e.g.
queries against an already-sketched dataset).

# Arguments
- `method`: `:gaussian` or `:qr`, selecting which of [`gaussian`](@ref)/[`qr`](@ref)
  builds the rotation matrix; any other value raises `ArgumentError`
- `outdim`: the number of sketch bits (i.e., the output dimension of the rotation)
- `data`: the vector or matrix (one vector per column) to sketch
- `rng`, `FloatType`: forwarded to the chosen rotation-matrix generator
- `minbatch`: (matrix method only) minimum number of columns processed per parallel task

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 128, 1000);

julia> B, R = Projections.bitsketch(:gaussian, 256, X);

julia> size(B), eltype(B)  # (4, 1000), UInt64

julia> q = rand(Float32, 128);

julia> bq = Projections.bitsketch(R, q);  # reuse R, comparable to B's columns
```
"""
function bitsketch(method::Symbol, outdim::Int, data::AbstractVecOrMat{<:AbstractFloat};
        rng::AbstractRNG=Random.default_rng(), FloatType::Type=Float32, minbatch::Int=4)
    indim = size(data, 1)
    rp = if method === :gaussian
        gaussian(rng, FloatType, indim, outdim)
    elseif method === :qr
        qr(rng, FloatType, indim, outdim)
    else
        throw(ArgumentError("bitsketch: unknown rotation method :$method (expected :gaussian or :qr)"))
    end

    B = data isa AbstractMatrix ? _bitsketch_apply(rp, data; minbatch) : _bitsketch_apply(rp, data)
    B, getmap(rp)
end

"""
    bitsketch(hp::HadamardProjection, v::AbstractVector{<:AbstractFloat}) -> Vector{UInt64}
    bitsketch(hp::HadamardProjection, X::AbstractMatrix{<:AbstractFloat}; minbatch::Int=4) -> Matrix{UInt64}

Computes a bit sketch the same way as [`bitsketch(R::AbstractMatrix, data)`](@ref), but
using the fast Walsh-Hadamard transform ([`HadamardProjection`](@ref)) instead of a dense
random rotation matrix: it encodes the sign of each of the `indim(hp)` transformed
coordinates into `UInt64`-packed bits (see [`packsigns`](@ref)).

!!! warning
    [`HadamardProjection`](@ref) requires `indim` to be a **power of two** (`fwht`'s own
    restriction); constructing `hp = HadamardProjection(indim)` with a non-power-of-two
    `indim` raises `ArgumentError`. Pad `v`/`X` with extra coordinates/rows to the next
    power of two beforehand if needed. As with the random-rotation forms, reuse the same
    `hp` to sketch a query and the dataset it will be compared against, so their sketches
    are comparable.

# Arguments
- `hp`: the [`HadamardProjection`](@ref) to apply
- `v`/`X`: the vector, or matrix (one vector per column), to sketch
- `minbatch`: (matrix method only) minimum number of columns processed per parallel task

# Examples

```julia
julia> using SimilaritySearch

julia> hp = Projections.HadamardProjection(128);  # 128 is a power of two

julia> X = rand(Float32, 128, 1000);

julia> B = Projections.bitsketch(hp, X);

julia> size(B), eltype(B)  # (2, 1000), UInt64  -- cld(128, 64) == 2 words per sketch
```
"""
bitsketch(hp::HadamardProjection, v::AbstractVector{<:AbstractFloat}) = _bitsketch_apply(hp, v)
bitsketch(hp::HadamardProjection, X::AbstractMatrix{<:AbstractFloat}; minbatch::Int=4) = _bitsketch_apply(hp, X; minbatch)
