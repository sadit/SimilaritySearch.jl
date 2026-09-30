# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, Random, Statistics

@testset "scores: recall and match error per query and macro, and their bootstrap (#92)" begin
    rng = Xoshiro(5)
    dim, n, nq, k = 8, 2000, 200, 10
    X = MatrixDatabase(rand(rng, Float32, dim, n))
    Q = MatrixDatabase(rand(rng, Float32, dim, nq))
    dist = Dist.SqL2()
    E = ExhaustiveSearch(dist, X)
    ctx = GenericContext()
    goldI, goldD = searchbatch(E, ctx, Q, k)
    exact = [search(E, ctx, Q[i], knnqueue(KnnSorted, k)) for i in 1:nq]
    G = SearchGraph(dist, X)
    gctx = SearchGraphContext(; reporters=[])
    index!(G, gctx)
    G.algo[] = BeamSearch(; bsize=2, Δ=1.0)                     # deliberately weak, so the scores have spread
    knns = [search(G, gctx, Q[i], knnqueue(KnnSorted, k)) for i in 1:nq]
    resI = reduce(hcat, [collect(IdView(r)) for r in knns])
    golddists = [goldD[:, i] for i in 1:nq]

    @testset "per-query scores are what the macro scores average" begin
        pq = perqueryscores(recallscore, goldI, resI)
        @test length(pq) == nq && all(0 .<= pq .<= 1)
        @test mean(pq) ≈ macrorecall(goldI, resI)
        @test mean(perqueryscores(recallscore, goldI, resI; k=5)) ≈ macrorecall(goldI, resI, 5)
        goldlist = [Set(goldI[:, i]) for i in 1:nq]
        @test perqueryscores(recallscore, goldlist, knns) == pq          # vectors of sets and queues, same values
        @test_throws DimensionMismatch perqueryscores(recallscore, goldI, resI[:, 1:10])
    end

    @testset "matcherror and macromatcherror, outside the optimizer" begin
        @test macromatcherror(goldD, exact) == 0.0                        # the exact result matches its own gold
        m = macromatcherror(goldD, knns)
        @test m > 0
        @test m ≈ mean(matcherror(goldD[:, i], knns[i], 1, 1) for i in 1:nq)
        @test macromatcherror(golddists, knns) == m                       # matrix or vector of gold distances
        @test macromatcherror(goldD, knns, MaxMatchError()) == m          # the parameters from the error function
        @test matcherror(goldD[:, 1], knns[1], MaxMatchError(; p=2f0)) == matcherror(goldD[:, 1], knns[1], 2, 1)
        @test macromatcherror(goldD, knns, 2, 1) < m                       # p = 2 suppresses the small deviations
        @test mean(perqueryscores((g, r) -> matcherror(g, r, 1, 1), goldD, knns)) ≈ m
        @test_throws DimensionMismatch macromatcherror(goldD, knns[1:3])
    end

    @testset "bootstrapscore: the distribution of a macro score over resampled queries" begin
        b = bootstrapscore(ones(100); rng)
        @test b.mean == 1 && b.std == 0 && b.lo == b.hi == 1 && b.level == 0.95
        @test length(b.samples) == 1000 && b.perquery == ones(100)
        pq = Float64.(rand(rng, 400) .< 0.7)                                 # Bernoulli(0.7): std of the mean is known
        b = bootstrapscore(pq; nboot=4000, rng)
        @test b.mean == mean(pq)
        expected = sqrt(mean(pq) * (1 - mean(pq)) / length(pq))
        @test abs(b.std - expected) < 0.2 * expected
        @test b.lo < b.mean < b.hi
        @test abs(b.lo - quantile(b.samples, 0.025)) < 1e-12 && abs(b.hi - quantile(b.samples, 0.975)) < 1e-12
        narrow = bootstrapscore(pq; nboot=4000, level=0.5, rng)
        @test narrow.hi - narrow.lo < b.hi - b.lo                            # a lower level, a narrower interval
        @test bootstrapscore(pq; rng=Xoshiro(9)).samples == bootstrapscore(pq; rng=Xoshiro(9)).samples   # reproducible
        @test bootstrapscore(pq; rng=Xoshiro(9)).samples != bootstrapscore(pq; rng=Xoshiro(10)).samples
        @test startswith(sprint(show, b), "BootstrapScore(")
        @test_throws ArgumentError bootstrapscore(Float64[])
        @test_throws ArgumentError bootstrapscore(pq; level=1.0)
        @test_throws ArgumentError bootstrapscore(pq; nboot=1)
        # through the scores
        br = bootstrapscore(recallscore, goldI, resI; rng)
        @test br.mean ≈ macrorecall(goldI, resI) && length(br.perquery) == nq
        @test bootstrapscore(recallscore, goldI, resI; k=5, rng).mean ≈ macrorecall(goldI, resI, 5)
        bm = bootstrapscore((g, r) -> matcherror(g, r, MaxMatchError()), goldD, knns; rng)
        @test bm.mean ≈ macromatcherror(goldD, knns)
        # paired: the weak graph against the exact result on the same queries never wins
        d = bootstrapscore(perqueryscores(recallscore, goldI, resI) .- perqueryscores(recallscore, goldI, goldI); rng)
        @test d.mean ≈ macrorecall(goldI, resI) - 1 && d.hi <= 0
        @info "scores over $nq queries: recall $(round(br.mean; digits=3)) ± $(round(br.std; digits=3)) [$(round(br.lo; digits=3)), $(round(br.hi; digits=3))]; match error $(round(bm.mean; digits=4)) ± $(round(bm.std; digits=4))"
    end
end
