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

    @testset "two disjoint exact indices" begin
        idxA, idxB = ExhaustiveSearch(dist, A), ExhaustiveSearch(dist, B)
        ctxA, ctxB = GenericContext(), GenericContext()
        i, j, d = bichromatic_closestpair(idxA, ctxA, idxB, ctxB)
        @test (i, j) == (gi, gj)
        @test d ≈ gd atol=1e-5
    end

    @testset "dataset wrapper (exact)" begin
        i, j, d = bichromatic_closestpair(dist, A, B)
        @test (i, j) == (gi, gj)
        @test d ≈ gd atol=1e-5
    end

    @testset "same index passed twice matches closestpair" begin
        idx = ExhaustiveSearch(dist, A)
        ctx = GenericContext()
        i1, j1, d1 = closestpair(idx, ctx)
        i2, j2, d2 = bichromatic_closestpair(idx, ctx, idx, ctx)
        @test (i1, j1, d1) == (i2, j2, d2)
    end

    @testset "distinct indices over the same database exclude self-matches" begin
        # two separately-built SearchGraphs over the same (by-reference) database A:
        # ExhaustiveSearch is an immutable, field-equal struct in this case, so two
        # separate ExhaustiveSearch(dist, A) calls would actually be `===` to each other
        # and wouldn't exercise this branch -- SearchGraph's mutable internal adjacency
        # makes idxA and idxB genuinely distinct objects here.
        idxA = SearchGraph(dist, A)
        idxB = SearchGraph(dist, A)  # same underlying database object, distinct index
        ctxA, ctxB = SearchGraphContext(), SearchGraphContext()
        index!(idxA, ctxA); index!(idxB, ctxB)
        @test database(idxA) === database(idxB)
        @test idxA !== idxB
        i, j, d = bichromatic_closestpair(idxA, ctxA, idxB, ctxB)
        @test i != j
        @test d > 0
    end
end
