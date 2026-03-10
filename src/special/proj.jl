module Projections

using Polyester, Random, LinearAlgebra, Distributions, StatsBase
export RandomProjections, outdim, indim, transform, transform!
using ...SimilaritySearch.Dist.CastF32: dot32

struct RandomProjections{M<:AbstractMatrix}
    map::M
end

getmap(rp::RandomProjections) = rp.map

function gaussian(rng::AbstractRNG, FloatType::Type, indim::Int, outdim::Int)
    N = Normal(zero(FloatType), convert(FloatType, 1 / outdim))
    M = rand(rng, N, indim, outdim)
    for c in eachcol(M)
        normalize!(c)
    end

    RandomProjections(M)
end

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
indim(rp::RandomProjections) = size(getmap(rp), 1)
outdim(rp::RandomProjections) = size(getmap(rp), 2)
Base.eltype(rp::RandomProjections) = eltype(getmap(rp))

function transform!(rp::RandomProjections, out::AbstractVector, v::AbstractVector)
    for (i, x) in enumerate(eachcol(getmap(rp)))
        @inbounds out[i] = dot32(x, v)
    end

    out
end

function transform(rp::RandomProjections, v::AbstractVector)
    out = Vector{eltype(rp)}(undef, outdim(rp))
    transform!(rp, out, v)
end

function transform(rp::RandomProjections, X::AbstractMatrix; minbatch::Int=4)
    O = Matrix{eltype(rp)}(undef, outdim(rp), size(X, 2))
    transform!(rp, O, X; minbatch)
end

function transform!(rp::RandomProjections, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)

    @batch per = thread minbatch = minbatch for i in 1:n
        o = view(O, :, i)
        x = view(X, :, i)
        transform!(rp, o, x)
    end

    O
end

end