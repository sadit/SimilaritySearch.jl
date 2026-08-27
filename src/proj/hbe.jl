# This file is a part of SimilaritySearch.jl

export RandomHyperplanes

"""
    RandomHyperplanes(dist::SemiMetric, refs::AbstractDatabase, npairs::Integer)

Binary sketch generator for a generic metric space, based on `npairs` pairs of reference
objects (anchors) drawn from `refs` -- so `refs` must hold exactly `2npairs` objects, laid
out as consecutive pairs (`refs[2i-1], refs[2i]` is the `i`-th pair). An object is encoded
into `npairs` bits: bit `i` is `1` when the object is closer to `refs[2i-1]` than to
`refs[2i]` (see [`bitsketch`](@ref)).

Unlike [`DistantHyperplanes`](@ref), which searches for and filters informative,
mutually-diverse anchor pairs from data, `RandomHyperplanes` takes the anchor pairs as
given -- e.g., a plain random sample of the dataset -- making it a much cheaper, simpler
baseline.

Sketches are compared with [`distance`](@ref)`(B)` (Hamming distance over `UInt64`-packed
bits).

# Arguments
- `dist`: the distance function of the underlying metric space
- `refs`: the `2npairs` reference objects, laid out as consecutive pairs
- `npairs`: the number of output bits (i.e., reference pairs); must be a multiple of 64

# Examples

```julia
julia> using SimilaritySearch

julia> X = MatrixDatabase(rand(Float32, 8, 10_000));

julia> refs = SubDatabase(X, rand(1:10_000, 256));  # 128 pairs -> 128 bits

julia> m = SimilaritySearch.Projections.RandomHyperplanes(SimilaritySearch.Dist.L2(), refs, 128);

julia> B = SimilaritySearch.Projections.bitsketch(m, X);

julia> size(B.matrix), eltype(B.matrix)  # (2, 10000), UInt64 -- 128 bits / 64 = 2 words per sketch
```
"""
struct RandomHyperplanes{DistType<:SemiMetric,DbType<:AbstractDatabase}
    dist::DistType
    refs::DbType
end

distance(::RandomHyperplanes) = Hamming()

"""
    outdim(B::RandomHyperplanes)

Returns the number of output bits (reference pairs) of `B`.
"""
outdim(B::RandomHyperplanes) = length(B.refs) ÷ 2
_hbe_nblocks(B::RandomHyperplanes) = outdim(B) ÷ 64

function RandomHyperplanes(dist::SemiMetric, refs::AbstractDatabase, npairs::Integer)
    npairs % 64 == 0 || throw(ArgumentError("npairs must be a factor of 64"))
    length(refs) == 2npairs || throw(ArgumentError("refs must contain 2*npairs elements"))
    RandomHyperplanes(dist, refs)
end

@inline function _hbe_side(B::RandomHyperplanes, v, i::Integer)::Bool
    i = 2i
    evaluate(B.dist, B.refs[i-1], v) < evaluate(B.dist, B.refs[i], v)
end

function _hbe_encode!(B::RandomHyperplanes, vout::AbstractVector{UInt64}, v)
    for w in eachindex(vout)
        base = (w - 1) * 64
        E = zero(UInt64)
        for b in 1:64
            _hbe_side(B, v, base + b) && (E |= one(UInt64) << (b - 1))
        end
        vout[w] = E
    end

    vout
end

"""
    bitsketch(B::RandomHyperplanes, v) -> Vector{UInt64}
    bitsketch(B::RandomHyperplanes, X::AbstractDatabase; minbatch::Int=4) -> MatrixDatabase

Encodes `v` (or every object of `X`) with the reference pairs of `B` (see
[`RandomHyperplanes`](@ref)), packed into `UInt64` words. Sketches are compared with
[`distance`](@ref)`(B)`.

# Arguments
- `B`: the `RandomHyperplanes` sketch generator
- `v`/`X`: the object, or database of objects, to sketch
- `minbatch`: (database method only) minimum number of items processed per parallel task
"""
bitsketch(B::RandomHyperplanes, v) = _hbe_encode!(B, Vector{UInt64}(undef, _hbe_nblocks(B)), v)

function bitsketch(B::RandomHyperplanes, X::AbstractDatabase; minbatch::Int=4)
    D = Matrix{UInt64}(undef, _hbe_nblocks(B), length(X))

    @BATCHES minbatch for i in 1:length(X)
        _hbe_encode!(B, view(D, :, i), X[i])
    end

    MatrixDatabase(D)
end
