# This file is a part of SimilaritySearch.jl
#
# The rotations an estimator lives in (ScalarQuant.SQEncoder, RaBitQ): the objects, and the
# `rotate`/`rotationdim` interface SimilaritySearch defines for them.

export RandomizedHadamard, Rotation
import ..SimilaritySearch: rotate, rotationdim

"""
    RandomizedHadamard(dim; rng=Random.default_rng())

A random orthogonal rotation computed in `dim log dim` flops: a random sign flip of every
coordinate followed by the Walsh-Hadamard transform (`Projections.HadamardProjection`, which
scales by `1/dim`), rescaled by `sqrt(dim)` so that norms are preserved. `dim` must be a power
of two. The plain transform is not a random rotation (a constant vector lands on a single
coordinate), and the sign flip is what makes it one.
"""
struct RandomizedHadamard
    hp::HadamardProjection
    signs::Vector{Float32}
    scale::Float32
end

function RandomizedHadamard(dim::Integer; rng::AbstractRNG=Random.default_rng())
    RandomizedHadamard(HadamardProjection(Int(dim)), rand(rng, (-1f0, 1f0), Int(dim)), Float32(sqrt(dim)))
end

"""
    Rotation

What the estimators take as their rotation: a [`RandomProjections`](@ref) with a square map,
such as `Projections.qr(dim, dim)` (an orthogonal matrix, `dim²` floats and `dim²` flops per
vector), or a [`RandomizedHadamard`](@ref) (`dim` a power of two, `dim` signs stored,
`dim log dim` flops). Both implement `rotate(rot, v)` and `rotationdim(rot)`;
`ScalarQuant.SQEncoder` also takes `nothing`, which rotates nothing.
"""
const Rotation = Union{RandomProjections,RandomizedHadamard}

"Rotates `v` with the orthogonal map, a fresh `Vector{Float32}` of the same norm."
rotate(rot::RandomProjections, v::AbstractVector) = transform(rot, v)

function rotate(rot::RandomizedHadamard, v::AbstractVector)
    out = Vector{Float32}(undef, length(rot.signs))
    @inbounds for i in eachindex(out)
        out[i] = rot.signs[i] * Float32(v[i])
    end
    transform!(rot.hp, out, out)
    out .*= rot.scale
    out
end

function rotationdim(rot::RandomProjections)
    d = indim(rot)
    d == outdim(rot) ||
        throw(ArgumentError("a $d -> $(outdim(rot)) projection is not a rotation; give a square map such as Projections.qr($d, $d)"))
    d
end
rotationdim(rot::RandomizedHadamard) = length(rot.signs)
