# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test, Random
Random.seed!(0)

@isdefined(FAST_TESTS) || (const FAST_TESTS = get(ENV, "FAST_TESTS", "false") == "true")

@testset "SpatialAccessTree approximate variants" begin
    dim, n, nq, k = 4, (FAST_TESTS ? 500 : 2_000), 30, 10
    dist = Dist.L2()
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, nq))

    gctx = GenericContext()
    seq = ExhaustiveSearch(dist, db)
    gold_ids, _ = searchbatch(seq, gctx, queries, k)

    sat = Sat(db; dist)
    index!(sat, GenericContext())

    ctx = SatContext()
    # optimize_index! runs a SearchModels autotuning search whose cost scales with
    # population/iterations, independently of n -- shrink both under FAST_TESTS.
    optkwargs = FAST_TESTS ? (; numqueries=16, initialpopulation=4, maxiters=4) : (;)

    for (build, minrecall) in [
        (s -> BeamSearchSat(s), 0.7),
        (s -> PruningSat(s), 0.6),
    ]
        idx = build(sat)
        optimize_index!(idx, ctx, MinRecall(0.9); optkwargs...)
        ids, _ = searchbatch(idx, ctx, queries, k)
        recall = macrorecall(gold_ids, ids)
        @test recall >= minrecall
    end

    # allknn-oriented variants: query id == database id
    gold_all_ids, _ = allknn(seq, gctx, k)

    for build in (s -> PrunParSat(s), s -> BeamSearchParSat(s))
        idx = build(sat)
        optimize_index!(idx, ctx, MinRecall(0.85); optkwargs...)
        all_ids, _ = allknn(idx, ctx, k)
        recall = macrorecall(gold_all_ids, all_ids)
        @test recall >= 0.5
    end

    # forest search
    parts = [index!(Sat(db; dist, root=r), GenericContext()) for r in (1, div(n, 3), 2 * div(n, 3))]
    midx = BeamSearchMultiSat(parts)
    optimize_index!(midx, ctx, MinRecall(0.9); optkwargs...)
    m_ids, _ = searchbatch(midx, ctx, queries, k)
    @test macrorecall(gold_ids, m_ids) >= 0.6
end
