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
    check_selection(R, X; nearest=true)

Everything a `CenterSelection` promises, verified against `X` itself rather than against the
algorithm that produced it. `nearest=false` drops the one promise `dnet` deliberately does not
make -- that the center an object is assigned to is the closest one -- while still requiring
`assigndist` to agree with `assign`.
"""
function check_selection(R::CenterSelection, X; nearest::Bool=true)
    n, k = length(X), length(R.centers)
    @test allunique(R.centers)
    @test all(c -> 1 <= c <= n, R.centers)
    @test length(R.assign) == n
    @test length(R.assigndist) == n

    # `assign` indexes `centers`, not `X`
    @test all(a -> 1 <= a <= k, R.assign)

    # every object really is at `assigndist` from the center `assign` names -- and, where the
    # selector promises it, no other center is closer
    D = [SimilaritySearch.evaluate(DIST, X[i], X[c]) for i in 1:n, c in R.centers]
    for i in 1:n
        @test R.assigndist[i] ≈ D[i, R.assign[i]]
        nearest && @test R.assigndist[i] ≈ minimum(view(D, i, :))
        # even without the nearest promise, an assignment can never claim to be closer than
        # the closest center actually is
        @test R.assigndist[i] >= minimum(view(D, i, :)) - 1f-6
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
        check_selection(R, X; nearest=false)

        # the count is exact, just not `numcenters`: balls of `n ÷ k` objects, carved until
        # nothing is left
        @test length(R.centers) == cld(n, n ÷ k)

        # and the documented exception is real, not theoretical: some object is filed under a
        # center that is not its closest one
        D = [SimilaritySearch.evaluate(DIST, X[i], X[c]) for i in 1:n, c in R.centers]
        @test any(i -> R.assigndist[i] > minimum(view(D, i, :)) + 1f-6, 1:n)
    end

    @testset "dnet picks its centers without an identifier bias" begin
        # the compaction used to sort the surviving pool, which walked large identifiers toward
        # the end of it -- and the end is exactly where the next center is taken from. Over 20
        # runs the mean selected identifier sat at 71% of the range instead of 50%
        m = 600
        Y = MatrixDatabase(rand(Float32, dim, m))
        acc = Float64[]
        for _ in 1:20
            R = dnet(DIST, Y, 12; verbose=false)
            push!(acc, sum(Int, R.centers) / length(R.centers))
        end

        @test 0.42m <= sum(acc) / length(acc) <= 0.58m
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
