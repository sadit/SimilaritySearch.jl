# This file is a part of SimilaritySearch.jl
using SimilaritySearch, Test, LinearAlgebra, SparseArrays, Random

const Spherical = SimilaritySearch.Special.Spherical
const Sparse = SimilaritySearch.Special.Sparse

@testset "Spherical: dense embedding" begin
    rng = MersenneTwister(1)
    X = rand(rng, Float32, 8, 200)
    se = Spherical.SphericalEmbedding(X)
    @test Spherical.indim(se) == 8
    @test Spherical.outdim(se) >= 9
    @test se.maxnorm ≈ maximum(norm(view(X, :, j)) for j in 1:size(X, 2)) atol=1f-4

    P = Spherical.transform(se, X)
    @test size(P) == (Spherical.outdim(se), size(X, 2))
    for j in 1:size(P, 2)
        @test norm(view(P, :, j)) ≈ 1f0 atol=1f-4
    end

    q = rand(rng, Float32, 8)
    Qq = Spherical.transform_query(se, q)
    @test length(Qq) == Spherical.outdim(se)
    @test norm(Qq) ≈ 1f0 atol=1f-4

    # ranking by inner product must be preserved by the embedding
    dots_orig = [dot(view(X, :, j), q) for j in 1:size(X, 2)]
    dots_emb = [dot(view(P, :, j), Qq) for j in 1:size(P, 2)]
    @test sortperm(dots_orig; rev=true) == sortperm(dots_emb; rev=true)

    se_nopad = Spherical.SphericalEmbedding(X; pad=false)
    @test Spherical.outdim(se_nopad) == 9

    # MatrixDatabase form matches the plain-matrix form
    db = MatrixDatabase(X)
    Pdb = Spherical.transform(se, db)
    @test Pdb.matrix == P
end

@testset "Spherical: sparse embedding (SparseMatrixCSC / SparseVector)" begin
    rng = MersenneTwister(2)
    Xs = sprand(rng, Float32, 12, 150, 0.3)
    se = Spherical.SphericalEmbedding(Xs)
    @test se.pad == 0
    @test Spherical.outdim(se) == 13

    Ps = Spherical.transform(se, Xs)
    @test size(Ps) == (Spherical.outdim(se), size(Xs, 2))
    for j in 1:size(Ps, 2)
        @test norm(view(Matrix(Ps), :, j)) ≈ 1f0 atol=1f-4
    end

    q = sprand(rng, Float32, 12, 0.3)
    Qq = Spherical.transform_query(se, q)
    @test norm(Vector(Qq)) ≈ 1f0 atol=1f-4

    dots_orig = [dot(Xs[:, j], q) for j in axes(Xs, 2)]
    dots_emb = [dot(Ps[:, j], Qq) for j in axes(Ps, 2)]
    @test sortperm(dots_orig; rev=true) == sortperm(dots_emb; rev=true)

    se_pad = Spherical.SphericalEmbedding(se.maxnorm, se.indim, 1)  # sparse inputs must have pad == 0
    @test_throws ArgumentError Spherical.transform(se_pad, q)
    @test_throws ArgumentError Spherical.transform_query(se_pad, q)
end

@testset "Spherical: Special.Sparse (SparseVecView / SparseDatabase)" begin
    rng = MersenneTwister(3)
    Xs = sprand(rng, Float32, 10, 80, 0.3)
    db = Sparse.SparseDatabase(Xs)
    se = Spherical.SphericalEmbedding(db)
    @test se.maxnorm ≈ Spherical.SphericalEmbedding(Xs).maxnorm atol=1f-5

    Pdb = Spherical.transform(se, db)
    @test length(Pdb) == length(db)
    for i in 1:length(db)
        v = Pdb[i]
        @test norm(collect(v.nzval)) ≈ 1f0 atol=1f-4
    end

    q = db[1]
    Qq = Spherical.transform_query(se, q)
    @test norm(collect(Qq.nzval)) ≈ 1f0 atol=1f-4
end

@testset "Spherical: stale maxnorm clamps instead of erroring" begin
    se = Spherical.SphericalEmbedding(2f0, 4, 0)
    xbig = fill(10f0, 4)  # norm(xbig) = 20 > maxnorm = 2 -- embedding is stale for this vector
    out = Spherical.transform(se, xbig)
    @test out[end] == 0f0  # clamped, not NaN/DomainError
end

@testset "Spherical: constructor rejects non-positive maxnorm" begin
    @test_throws ArgumentError Spherical.SphericalEmbedding(0f0, 4, 0)
    @test_throws ArgumentError Spherical.SphericalEmbedding(zeros(Float32, 4, 10))
end
