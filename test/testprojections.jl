# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, SimilaritySearch.Projections, LinearAlgebra, Random

@testset "Projections and BitSketches" begin
    Random.seed!(42)
    dim = 128
    n = 20
    X = randn(Float32, dim, n)
    v = randn(Float32, dim)

    @testset "HadamardProjection" begin
        hp = HadamardProjection(dim)
        @test indim(hp) == dim
        @test outdim(hp) == dim
        @test size(hp) == (dim, dim)
        @test_throws ArgumentError HadamardProjection(100) # not a power of 2
        @test_throws ArgumentError HadamardProjection(dim, 64) # outdim != indim

        # Single vector transform
        y = transform(hp, v)
        @test length(y) == dim
        @test eltype(y) == Float32

        # In-place transform!
        y_in = similar(v)
        transform!(hp, y_in, v)
        @test y_in ≈ y

        # In-place on self
        v_copy = copy(v)
        transform!(hp, v_copy, v_copy)
        @test v_copy ≈ y

        # Matrix transform
        Y = transform(hp, X)
        @test size(Y) == size(X)
        @test eltype(Y) == Float32

        # Matrix in-place
        Y_in = similar(X)
        transform!(hp, Y_in, X)
        @test Y_in ≈ Y

        for i in 1:n
            @test Y[:, i] ≈ transform(hp, X[:, i])
        end
    end

    @testset "RandomProjections" begin
        out_d = 32
        rp_gauss = Projections.gaussian(dim, out_d)
        @test indim(rp_gauss) == dim
        @test outdim(rp_gauss) == out_d

        y = transform(rp_gauss, v)
        @test length(y) == out_d

        Y = transform(rp_gauss, X)
        @test size(Y) == (out_d, n)

        rp_qr = Projections.qr(dim, out_d)
        @test indim(rp_qr) == dim
        @test outdim(rp_qr) == out_d
    end

    @testset "BitSketches" begin
        hp = HadamardProjection(dim)
        b_vec = bitsketch(hp, v)
        @test length(b_vec) == cld(dim, 64)
        @test eltype(b_vec) == UInt64

        B_mat = bitsketch(hp, X)
        @test size(B_mat) == (cld(dim, 64), n)
        @test eltype(B_mat) == UInt64
    end
end
