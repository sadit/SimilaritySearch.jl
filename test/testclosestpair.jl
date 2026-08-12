# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra, Random
Random.seed!(0)

@testset "closestpair" begin
    dist = SimilaritySearch.Dist.Cosine()
    dim, mindist = 2, 1e-4
    db = MatrixDatabase(rand(Float32, dim, 1000))
    G = SearchGraph(dist, db)
    ctx = SearchGraphContext()
    tG = @elapsed index!(G, ctx)
    tG += @elapsed i, j, d = closestpair(G, ctx)
    @test i != j
    @test d < mindist
    @show i, j, d
    i, j, d = closestpair(G, ctx)
    @test i != j
    @test d < mindist
    @show i, j, d, :parallel
    seq = ExhaustiveSearch(dist, db)
    ctxseq = GenericContext()
    tE = @elapsed i, j, d = closestpair(seq, ctxseq)
    @info "NOTE: the exact method will be faster on small datasets due to the preprocessing step of the approximation method"
    @info "closestpair computation time", :approx => tG, :exact => tE

    # @test_call closestpair(G, ctx; minbatch=-1)
end

@testset "bichromatic_closestpair" begin
    dist = SimilaritySearch.Dist.SqL2()
    dim = 2
    A = MatrixDatabase(rand(Float32, dim, 60))
    B = MatrixDatabase(rand(Float32, dim, 80))

    function bruteforce(A, B)
        bi, bj, bd = 0, 0, typemax(Float32)
        for i in eachindex(A), j in eachindex(B)
            d = SimilaritySearch.Dist.evaluate(dist, A[i], B[j])
            if d < bd
                bi, bj, bd = i, j, d
            end
        end

        bi, bj, bd
    end

    gi, gj, gd = bruteforce(A, B)

    @testset "exact index over A, raw database B" begin
        idxA = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        @test database(idxA) !== B  # disjoint datasets -> samedata defaults to false
        i, j, d = bichromatic_closestpair(idxA, ctx, B)
        @test (i, j) == (gi, gj)
        @test d ≈ gd atol=1e-5
    end

    @testset "dataset wrapper (exact)" begin
        i, j, d = bichromatic_closestpair(dist, A, B)
        @test (i, j) == (gi, gj)
        @test d ≈ gd atol=1e-5
    end

    @testset "dataset wrapper (approximate, SearchGraph)" begin
        i, j, d = bichromatic_closestpair(dist, A, B; recall=0.9)
        @test d >= gd  # approximate: never better than the true minimum, may coincide or be worse
    end

    @testset "closestpair(idx, ctx) matches bichromatic_closestpair(idx, ctx, database(idx))" begin
        idx = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        i1, j1, d1 = closestpair(idx, ctx)
        i2, j2, d2 = bichromatic_closestpair(idx, ctx, database(idx))
        @test (i1, j1, d1) == (i2, j2, d2)
    end

    @testset "SearchGraph fast path when idxA indexes B itself (samedata auto-detected)" begin
        idxA = SearchGraph(dist, A)
        ctx = SearchGraphContext()
        index!(idxA, ctx)
        @test database(idxA) === A
        i, j, d = bichromatic_closestpair(idxA, ctx, A)
        @test i != j
        @test d > 0
    end

    @testset "explicit samedata override" begin
        # A2 has the same values as A but is a distinct object: without an override, every
        # element has a 0-distance "twin" in A2, which is the legitimate answer for two
        # datasets that just happen to coincide -- samedata=true forces self-exclusion instead.
        A2 = MatrixDatabase(copy(A.matrix))
        idxA = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        @test database(idxA) !== A2

        i0, j0, d0 = bichromatic_closestpair(idxA, ctx, A2)
        @test d0 == 0

        i1, j1, d1 = bichromatic_closestpair(idxA, ctx, A2; samedata=true)
        @test i1 != j1
        @test d1 > 0
    end
end

@testset "bichromatic_kclosestpairs (k >= 1)" begin
    dist = SimilaritySearch.Dist.SqL2()
    dim, k = 2, 5
    A = MatrixDatabase(rand(Float32, dim, 60))
    B = MatrixDatabase(rand(Float32, dim, 80))

    function bruteforce_topk(A, B, k; excludeself=false)
        all = Tuple{Int32,Int32,Float32}[]
        for i in eachindex(A), j in eachindex(B)
            (excludeself && i == j) && continue
            d = SimilaritySearch.Dist.evaluate(dist, A[i], B[j])
            push!(all, (Int32(i), Int32(j), Float32(d)))
        end

        sort!(all; by=last)
        all[1:min(k, length(all))]
    end

    gpairs = bruteforce_topk(A, B, k)

    @testset "exact index over A, raw database B" begin
        idxA = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        pairs = bichromatic_kclosestpairs(idxA, ctx, B; k)
        @test length(pairs) == k
        @test issorted(pairs; by=last)
        @test pairs == gpairs
    end

    @testset "k=1 matches bichromatic_closestpair" begin
        idxA = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        single = bichromatic_closestpair(idxA, ctx, B)
        pairs = bichromatic_kclosestpairs(idxA, ctx, B; k=1)
        @test length(pairs) == 1
        @test pairs[1] == single
    end

    @testset "dataset wrapper (exact)" begin
        pairs = bichromatic_kclosestpairs(dist, A, B; k)
        @test pairs == gpairs
    end

    @testset "closestpairs(idx, ctx) excludes self-matches" begin
        idx = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        pairs = closestpairs(idx, ctx; k)
        gself = bruteforce_topk(A, A, k; excludeself=true)
        # distance is symmetric, so (i, j) and (j, i) are exact ties -- compare unordered
        # pairs (canonicalized as (min, max)) instead of exact (i, j) tuple identity/order.
        canon(p) = (min(p[1], p[2]), max(p[1], p[2]), p[3])
        @test sort!(canon.(pairs)) == sort!(canon.(gself))
        @test all(i != j for (i, j, _) in pairs)
    end

    @testset "SearchGraph fast path when idxA indexes B itself" begin
        idxA = SearchGraph(dist, A)
        ctx = SearchGraphContext()
        index!(idxA, ctx)
        pairs = bichromatic_kclosestpairs(idxA, ctx, A; k)
        @test length(pairs) == k
        @test issorted(pairs; by=last)
        @test all(i != j for (i, j, _) in pairs)
    end

    @testset "fewer than k eligible pairs" begin
        tiny = MatrixDatabase(rand(Float32, dim, 3))
        idx = ExhaustiveSearch(dist, tiny)
        ctx = GenericContext()
        pairs = closestpairs(idx, ctx; k=100)  # only 3*2=6 ordered self-excluded pairs exist
        @test length(pairs) == 6
    end
end
