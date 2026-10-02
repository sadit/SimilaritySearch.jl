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

@testset "the goals' objective: a finite-support hinge on the target over the log cost" begin
    using SimilaritySearch: goalvalue
    g = MinRecall(0.9; tradeoff=1.5, width=0.02)
    rate = log(1.5) / 0.01
    @test goalvalue(g, 100, 0.95) == log(100)                                  # beyond the zone: the cost alone, exactly
    @test goalvalue(g, 100, 0.92) == log(100)                                  # one width above the target: still zero
    @test goalvalue(g, 100, 0.9) ≈ log(100) + rate * 0.02 / 4 rtol=1e-6        # at the target: width / 4 (the goal stores Float32)
    @test goalvalue(g, 100, 0.5) ≈ log(100) + rate * 0.4 rtol=1e-6             # far below: linear in the shortfall
    @test goalvalue(g, 200, 0.5) - goalvalue(g, 100, 0.5) ≈ log(2)             # and the cost still counts there
    vals = [goalvalue(g, 100, r) for r in 0.80:0.0005:1.0]
    @test issorted(vals; rev=true) && maximum(abs, diff(vals)) < 0.03         # decreasing in recall, no jump
    slopes = diff(vals) ./ 0.0005
    @test maximum(abs, diff(slopes)) < rate * 0.05                             # and no jump in the slope either (C¹)
    @test goalvalue(g, 50, 0.899) < goalvalue(g, 100, 0.95)                    # a hair below, at half the cost, wins
    h = MinRecall(0.9; tradeoff=1.5, width=1e-4)                               # a tiny width is the hard constraint
    @test goalvalue(h, 100, 0.95) == log(100)
    @test goalvalue(h, 100, 0.85) ≈ log(100) + rate * 0.05 rtol=1e-3
    # the transition zone, as multipliers of the width
    below = MinRecall(0.9; tradeoff=1.5, width=0.02, transition=(0, 2))
    above = MinRecall(0.9; tradeoff=1.5, width=0.02, transition=(-2, 0))
    @test goalvalue(below, 100, 0.9) == log(100)                               # (0, 2) charges nothing at the target
    @test goalvalue(below, 100, 0.89) > log(100) && goalvalue(below, 100, 0.89) < goalvalue(g, 100, 0.89)
    @test goalvalue(above, 100, 0.9) ≈ log(100) + rate * 0.02 rtol=1e-6        # (-2, 0) charges a full width at the target
    @test goalvalue(above, 100, 0.94) == log(100)                              # and nothing from two widths above
    @test goalvalue(below, 100, 0.5) ≈ goalvalue(g, 100, 0.5) - rate * 0.02 && goalvalue(above, 100, 0.5) ≈ goalvalue(g, 100, 0.5) + rate * 0.02
    hard = MinRecall(0.9; tradeoff=1.5, width=0.02, transition=(0, 0))         # lo == hi: a plain threshold at the target
    @test goalvalue(hard, 100, 0.9) == log(100)
    @test goalvalue(hard, 100, 0.89) ≈ log(100) + rate * 0.01 rtol=1e-6
    asym = MinRecall(0.9; tradeoff=1.5, width=0.02, transition=(-1, 3))        # any pair: still C¹, zero before, linear after
    avals = [goalvalue(asym, 100, r) for r in 0.80:0.0005:1.0]
    @test issorted(avals; rev=true) && maximum(abs, diff(diff(avals) ./ 0.0005)) < rate * 0.05
    @test goalvalue(asym, 100, 0.93) == log(100) && goalvalue(asym, 100, 0.5) ≈ goalvalue(g, 100, 0.5) - rate * 0.02
    # match error: the shortfall is the excess over maxerror
    e = MaxMatchError(; maxerror=0.1f0, tradeoff=2.0, width=0.01)
    @test goalvalue(e, 100, 0.05) == log(100)
    @test goalvalue(e, 100, 0.3) ≈ log(100) + log(2.0) / 0.01 * 0.2 rtol=1e-6
    @test goalvalue(e, 100, 0.0) < goalvalue(e, 101, 0.0)                     # increasing in the cost
    # constructors keep the positional target, validate the knobs, and leave the width to resolve
    @test MinRecall(0.95).minrecall == 0.95f0 && MinRecall(0.95).width === nothing && MinRecall(0.95).tradeoff == 1.5 && MinRecall(0.95).transition == (-1f0, 1f0)
    @test MinRecall(; minrecall=0.8, tradeoff=2, transition=(-2, 0)).transition == (-2f0, 0f0)
    @test_throws ArgumentError MinRecall(0.9; tradeoff=1.0)
    @test_throws ArgumentError MinRecall(0.9; tradeoff=Inf)
    @test_throws ArgumentError MinRecall(0.9; transition=(1, 0))               # lo > hi
    @test_throws ArgumentError MinRecall(0.9; transition=(-Inf, 1))
    @test_throws ArgumentError MaxMatchError(; width=0.0)
    @test_throws ArgumentError MaxMatchError(; transition=(2, 1))
    @test_throws ArgumentError goalvalue(MinRecall(0.9), 100, 0.95)            # unresolved width
    @test goalvalue(MinRecall(0.9), 100, 0.95; width=0.02) ≈ goalvalue(g, 100, 0.95)
    # through optimize_index!, with an explicit width, with the resolved one, and with other zones
    rng = Xoshiro(11)
    X = MatrixDatabase(rand(rng, Float32, 8, 2000)); Q = MatrixDatabase(rand(rng, Float32, 8, 64))
    G = SearchGraph(Dist.SqL2(), X); gctx = SearchGraphContext(; reporters=[]); index!(G, gctx)
    for kind in (MinRecall(0.8; width=0.05), MinRecall(0.8; transition=(0, 2)), MinRecall(0.8; transition=(-2, 0)), MaxMatchError(; maxerror=0.05f0))
        optimize_index!(G, gctx, kind; queries=Q)
        @test G.algo[] isa BeamSearch
    end
end
