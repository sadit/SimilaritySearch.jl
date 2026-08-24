# This file is a part of SimilaritySearch.jl

# Every algorithm in `Selection` returns an `AbstractSelection`, so all of them are held to the
# same invariants here -- against brute force rather than against the algorithm that produced the
# answer. The fixed-count selectors and `neardup` are duals (one fixes `k` and the radius falls
# out, the other fixes the radius and `k` falls out), and this file checks that they really do
# report the same things under the same names.
using Test, SimilaritySearch, LinearAlgebra
import SimilaritySearch: AbstractSelection, CenterSelection, NearDupSelection
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
function check_selection(R::AbstractSelection, X; nearest::Bool=true)
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

    # a center is covered by itself, at distance zero
    for (pos, c) in enumerate(R.centers)
        @test R.assign[c] == pos
        @test R.assigndist[c] ≈ 0
    end

    @test R.costdists >= 0
end

"the smallest distance between two of `ids`, the slow way"
function separation_of(X, ids)
    length(ids) < 2 && return typemax(Float32)
    P = pairdists(X, ids)
    k = length(ids)
    minimum(P[i, j] for i in 1:k, j in 1:k if i != j)
end

@testset "fixed-count selection: fft, dnet, randsel, multirandsel" begin
    n, dim, k = 120, 4, 10
    X = MatrixDatabase(rand(Float32, dim, n))

    @testset "fft" begin
        R = fft(DIST, X, k; verbose=false)
        @test length(R.centers) == k
        check_selection(R, X)
        @test R.separation ≈ separation_of(X, R.centers)
        @test R.costblocks == 0

        # the traversal's own guarantee: no object is farther from the selection than any two
        # centers are from each other
        @test R.covering <= R.separation
    end

    @testset "dnet" begin
        R = dnet(DIST, X, k; verbose=false)
        check_selection(R, X; nearest=false)
        @test R.separation ≈ separation_of(X, R.centers)

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
        @test R.separation ≈ separation_of(X, R.centers)
    end

    @testset "multirandsel" begin
        R = multirandsel(DIST, X, k)
        @test length(R.centers) == k
        check_selection(R, X)
        @test R.separation ≈ separation_of(X, R.centers)
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

@testset "radius-driven selection: neardup" begin
    n, dim, ϵ = 400, 4, 0.25f0
    X = MatrixDatabase(rand(Float32, dim, n))

    @testset "the dist-based wrapper" begin
        R = neardup(DIST, X, ϵ; blocksize=64, reporters=[])
        check_selection(R, X; nearest=false)   # incremental: see AbstractSelection
        @test R.costblocks >= 0

        # what makes it an ϵ-net, from both sides
        @test all(<=(R.epsilon), R.assigndist)
        @test R.covering <= R.epsilon
        @test separation_of(X, R.centers) > R.epsilon

        # the count is the output here, not the input
        @test 0 < length(R.centers) < n
        @test length(R.idx) == length(R.centers)

        # and the incremental exception is real: some object is covered by a center that is not
        # its closest one, because that center did not exist yet when the object was processed
        D = [SimilaritySearch.evaluate(DIST, X[i], X[c]) for i in 1:n, c in R.centers]
        @test any(i -> R.assigndist[i] > minimum(view(D, i, :)) + 1f-6, 1:n)
    end

    @testset "the index-based method, across block sizes" begin
        for blocksize in (16, 64, n), filterblocks in (true, false)
            G = SearchGraph(DIST, VectorDatabase(Vector{Float32}[]))
            ctx = SearchGraphContext(reporters=[])
            R = neardup(G, ctx, X, ϵ; blocksize, filterblocks)

            check_selection(R, X; nearest=false)
            @test all(<=(R.epsilon), R.assigndist)
            @test R.covering <= R.epsilon
            @test R.idx === G                     # the caller's own index, filled in place
            @test length(G) == length(R.centers)
        end
    end

    @testset "a tighter ϵ keeps more of the database" begin
        loose = neardup(DIST, X, 0.4f0; blocksize=64, reporters=[])
        tight = neardup(DIST, X, 0.1f0; blocksize=64, reporters=[])
        @test length(loose.centers) < length(tight.centers)
        @test loose.covering <= 0.4f0
        @test tight.covering <= 0.1f0
    end

    @testset "edge cases" begin
        # a negative ϵ used to be documented as selecting a quantile, and silently did nothing at
        # all: every distance exceeds it, so every object became its own center
        @test_throws ArgumentError neardup(DIST, X, -0.5; reporters=[])

        # ϵ = 0 is meaningful: exact duplicates only
        Y = MatrixDatabase(hcat(rand(Float32, dim, 10), rand(Float32, dim, 10)[:, 1:5]))
        R = neardup(DIST, Y, 0f0; blocksize=8, reporters=[])
        @test length(R.centers) == length(Y)      # nothing is an exact duplicate here
        @test R.covering == 0f0

        R = neardup(DIST, MatrixDatabase(rand(Float32, dim, 0)), ϵ; reporters=[])
        @test isempty(R.centers)
        @test isempty(R.assign)
    end
end
