# This file is a part of SimilaritySearch.jl

# Every center selector returns the same `CenterSelection`, so every one of them is held to
# the same invariants here -- and the two radii are checked against brute force, because they
# used to share a name (`ε`) while meaning different things in different selectors.
using Test, SimilaritySearch, LinearAlgebra
import SimilaritySearch: CenterSelection
using Random
Random.seed!(0)

const DIST = SimilaritySearch.Dist.L2()

"the distance between every pair of `X` objects named by `ids`, computed the slow, obvious way"
function pairdists(X, ids)
    [SimilaritySearch.evaluate(DIST, X[a], X[b]) for a in ids, b in ids]
end

"""
    check_selection(R, X)

Everything a `CenterSelection` promises, verified against `X` itself rather than against the
algorithm that produced it.
"""
function check_selection(R::CenterSelection, X)
    n, k = length(X), length(R.centers)
    @test allunique(R.centers)
    @test all(c -> 1 <= c <= n, R.centers)
    @test length(R.assign) == n
    @test length(R.assigndist) == n

    # `assign` indexes `centers`, not `X`
    @test all(a -> 1 <= a <= k, R.assign)

    # every object really is at `assigndist` from the center `assign` names, and no other
    # center is closer -- this is what makes the assignment an assignment
    D = [SimilaritySearch.evaluate(DIST, X[i], X[c]) for i in 1:n, c in R.centers]
    for i in 1:n
        @test R.assigndist[i] ≈ D[i, R.assign[i]]
        @test R.assigndist[i] ≈ minimum(view(D, i, :))
    end

    @test R.covering ≈ maximum(R.assigndist)

    if k >= 2
        P = pairdists(X, R.centers)
        @test R.separation ≈ minimum(P[i, j] for i in 1:k, j in 1:k if i != j)
    else
        @test R.separation == typemax(Float32)
    end

    @test R.costdists >= 0
    @test R.costblocks == 0
end

@testset "center selection: fft, dnet, randsel, multirandsel" begin
    n, dim, k = 120, 4, 10
    X = MatrixDatabase(rand(Float32, dim, n))

    @testset "fft" begin
        R = fft(DIST, X, k; verbose=false)
        @test length(R.centers) == k
        check_selection(R, X)

        # the traversal's own guarantee: no object is farther from the selection than any two
        # centers are from each other
        @test R.covering <= R.separation
    end

    @testset "dnet" begin
        R = dnet(DIST, X, k; verbose=false)
        check_selection(R, X)
        # `numcenters` is a target, not a promise -- pinned loosely on purpose, so that a
        # change in how the balls are carved shows up here instead of surprising a caller
        @test k <= length(R.centers) <= 2k
    end

    @testset "randsel" begin
        R = randsel(DIST, X, k)
        @test length(R.centers) == k
        check_selection(R, X)
    end

    @testset "multirandsel" begin
        R = multirandsel(DIST, X, k)
        @test length(R.centers) == k
        check_selection(R, X)
    end

    @testset "covering and separation are not the same number" begin
        # the reason they stopped sharing the name `ε`
        R = fft(DIST, X, k; verbose=false)
        @test R.covering != R.separation
    end

    @testset "identifiers are one indexing away from positions" begin
        R = fft(DIST, X, k; verbose=false)
        ids = R.centers[R.assign]
        @test length(ids) == length(X)
        @test Set(ids) ⊆ Set(R.centers)
        # a center belongs to its own basket
        for (pos, c) in enumerate(R.centers)
            @test R.assign[c] == pos
            @test R.assigndist[c] ≈ 0
        end
    end

    @testset "edge cases" begin
        for sel in (fft, dnet, randsel, multirandsel)
            @test_throws ArgumentError sel(DIST, X, 0)
        end

        empty_db = MatrixDatabase(rand(Float32, dim, 0))
        for sel in (fft, dnet, randsel, multirandsel)
            R = sel(DIST, empty_db, k)
            @test isempty(R.centers)
            @test isempty(R.assign)
        end

        # asking for more centers than there are objects returns each object once, never a
        # repeated center
        for R in (fft(DIST, X, n + 25; verbose=false), randsel(DIST, X, n + 25), multirandsel(DIST, X, n + 25))
            @test length(R.centers) == n
            @test allunique(R.centers)
        end

        R = fft(DIST, X, 1; verbose=false)
        @test length(R.centers) == 1
        check_selection(R, X)
    end

    @testset "a rejected assignment is caught at construction" begin
        # `assign` holding an identifier into X instead of a position is the mistake this type
        # exists to make impossible; with k=2 centers, position 3 cannot be valid
        @test_throws ArgumentError CenterSelection(UInt32[7, 9], UInt32[3], Float32[0.5], 0.5f0, 1.0f0, 0, 0)
        @test_throws ArgumentError CenterSelection(UInt32[7, 7], UInt32[1], Float32[0.5], 0.5f0, 1.0f0, 0, 0)
        @test_throws ArgumentError CenterSelection(UInt32[7], UInt32[1, 1], Float32[0.5], 0.5f0, 1.0f0, 0, 0)
    end
end
