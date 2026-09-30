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

        # the butterfly against the Sylvester Hadamard matrix, natural ordering, scaled by 1/n
        # (what Hadamard.jl's fwht_natural! produced; issue #89 replaced the FFTW plans)
        sylvester(k) = k == 0 ? ones(Float64, 1, 1) : (H = sylvester(k - 1); [H H; H -H])
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 1024), T in (Float32, Float64)   # scalar (< 8), one block (8), SIMD passes (>= 16)
            H = sylvester(Int(log2(m)))
            u = randn(T, m)
            want = T.(H * Float64.(u) ./ m)
            got = Projections.fwht!(copy(u))
            @test got ≈ want atol=(T == Float32 ? 1f-4 : 1e-10) * maximum(abs, want)
            @test got == transform(HadamardProjection(m), u)                   # the projection is the butterfly
            @test Projections.fwht!(Projections.fwht!(copy(u))) ≈ u ./ m       # H*H = n*I, twice scaled by 1/n
        end
        # any other layout or element type takes the scalar butterfly, and agrees bit for bit
        u = randn(Float32, 256)
        @test Projections.fwht!(view(copy(u), 1:1:256)) == Projections.fwht!(copy(u))          # a strided view: scalar path
        @test Projections.fwht!(Float64.(u)) ≈ Projections.fwht!(copy(u)) rtol=1f-5
        # the matrix path is the per-column butterfly, in parallel, and takes any batch size
        Y1 = transform(hp, X)
        @test Y1 == reduce(hcat, transform(hp, X[:, j]) for j in 1:n)
        for minbatch in (1, 3, 1000)
            O = similar(X)
            @test transform!(hp, O, X; minbatch) == Y1
        end
        Xw = randn(Float32, 256, 5_000)
        @test transform(Xw |> x -> HadamardProjection(256), Xw) == reduce(hcat, transform(HadamardProjection(256), Xw[:, j]) for j in 1:5_000)

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

    @testset "PCAProjection" begin
        Xpca = randn(Float32, dim, 2000)
        out_d = 16
        p = Projections.PCAProjection(Xpca, out_d)
        @test indim(p) == dim
        @test outdim(p) == out_d
        @test size(p) == (dim, out_d)

        y = transform(p, Xpca[:, 1])
        @test length(y) == out_d
        @test eltype(y) == Float32

        y_in = similar(y)
        transform!(p, y_in, Xpca[:, 1])
        @test y_in ≈ y

        Y = transform(p, Xpca)
        @test size(Y) == (out_d, size(Xpca, 2))

        Y_in = similar(Y)
        transform!(p, Y_in, Xpca)
        @test Y_in ≈ Y

        for i in 1:10
            @test Y[:, i] ≈ transform(p, Xpca[:, i])
        end

        b_vec = bitsketch(p, Xpca[:, 1])
        @test length(b_vec) == cld(out_d, 64)
        @test eltype(b_vec) == UInt64

        B_mat = bitsketch(p, Xpca)
        @test size(B_mat) == (cld(out_d, 64), size(Xpca, 2))
        @test eltype(B_mat) == UInt64
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

    @testset "RandomHyperplanes" begin
        db = MatrixDatabase(X)
        npairs = 64
        refs = SubDatabase(db, rand(1:n, 2npairs))
        m = RandomHyperplanes(SimilaritySearch.Dist.L2(), refs, npairs)
        @test outdim(m) == npairs

        b_vec = bitsketch(m, v)
        @test length(b_vec) == npairs ÷ 64
        @test eltype(b_vec) == UInt64

        B = bitsketch(m, db)
        @test size(B.matrix) == (npairs ÷ 64, n)
        @test eltype(B.matrix) == UInt64
        @test all(bitsketch(m, db[i]) == B[i] for i in 1:n)

        @test_throws ArgumentError RandomHyperplanes(SimilaritySearch.Dist.L2(), refs, npairs + 1) # not a factor of 64
        @test_throws ArgumentError RandomHyperplanes(SimilaritySearch.Dist.L2(), refs, 2npairs) # refs too short
    end

    @testset "DistantHyperplanes" begin
        db = MatrixDatabase(randn(Float32, dim, 2^12))
        nbits = 64
        m = DistantHyperplanes(SimilaritySearch.Dist.L2(), db, nbits; henc=2^9, verbose=false)
        @test outdim(m) == nbits

        b_vec = bitsketch(m, v)
        @test length(b_vec) == nbits ÷ 64
        @test eltype(b_vec) == UInt64

        B = bitsketch(m, db)
        @test size(B.matrix) == (nbits ÷ 64, length(db))
        @test eltype(B.matrix) == UInt64
        @test all(bitsketch(m, db[i]) == B[i] for i in 1:length(db))

        @test_throws ArgumentError DistantHyperplanes(SimilaritySearch.Dist.L2(), db, nbits + 1; henc=2^9) # not a factor of 64
    end

    @testset "AnchoredDistantHyperplanes" begin
        db = MatrixDatabase(randn(Float32, dim, 2^12))
        nbits = 64

        for (anchor, anchorpolicy) in ((nothing, :random), (nothing, :extremal), (1, :random), (db[3], :random))
            m = AnchoredDistantHyperplanes(SimilaritySearch.Dist.L2(), db, nbits; anchor, anchorpolicy, henc=2^9, verbose=false)
            @test outdim(m) == nbits

            b_vec = bitsketch(m, v)
            @test length(b_vec) == nbits ÷ 64
            @test eltype(b_vec) == UInt64

            B = bitsketch(m, db)
            @test size(B.matrix) == (nbits ÷ 64, length(db))
            @test eltype(B.matrix) == UInt64
            @test all(bitsketch(m, db[i]) == B[i] for i in 1:length(db))
        end

        @test_throws ArgumentError AnchoredDistantHyperplanes(SimilaritySearch.Dist.L2(), db, nbits; henc=2^9, anchorpolicy=:bogus)
        @test_throws ArgumentError AnchoredDistantHyperplanes(SimilaritySearch.Dist.L2(), db, nbits + 1; henc=2^9) # not a factor of 64
    end
end
