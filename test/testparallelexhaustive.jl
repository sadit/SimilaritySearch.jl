# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test, Random
import SimilaritySearch: ParallelExhaustiveSearch

Random.seed!(0)

@testset "ParallelExhaustiveSearch" begin
    dist = Dist.SqL2()
    n = 300
    dim = 6
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, 40))

    pex = ParallelExhaustiveSearch(dist, db)
    gold = ExhaustiveSearch(dist, db)
    ctx = GenericContext()
    gctx = GenericContext()  # gold's own context, kept separate from pex's

    @testset "single-query search(::ParallelExhaustiveSearch, ::GenericContext, q, ::AbstractKnnQueue)" begin
        # direct regression test for #53: this call shape threw MethodError because the
        # per-batch buffer was still a leftover Matrix{IdDist} (AoS) instead of matching
        # knnqueue's (ids, dists) SoA storage.
        for QueueType in (KnnSorted, KnnHeap), k in (1, 4, n, n + 5)
            for j in 1:length(queries)
                q = queries[j]
                r = knnqueue(QueueType, min(k, n))
                g = knnqueue(KnnSorted, min(k, n))
                search(pex, ctx, q, r)
                search(gold, gctx, q, g)
                @test length(r) == length(g)
                @test Set(IdView(r)) == Set(IdView(g))
                @test sort(collect(DistView(r))) ≈ sort(collect(DistView(g)))
            end
        end
    end

    @testset "searchbatch (matrix-based) matches ExhaustiveSearch" begin
        k = 5
        knns_ids, knns_dists = searchbatch(pex, ctx, queries, k; sorted=true)
        gold_ids, gold_dists = searchbatch(gold, gctx, queries, k; sorted=true)
        @test size(knns_ids) == (k, length(queries))
        @test macrorecall(gold_ids, knns_ids) == 1.0
        @test knns_dists ≈ gold_dists
    end

    @testset "searchbatch! with a vector of per-query KnnHeap containers" begin
        k = 6
        knns = [knnqueue(KnnHeap, k) for _ in 1:length(queries)]
        searchbatch!(pex, ctx, queries, knns)
        gold_ids, gold_dists = searchbatch(gold, gctx, queries, k; sorted=true)
        for (j, r) in enumerate(knns)
            @test Set(IdView(sortitems!(r))) == Set(gold_ids[:, j])
        end
    end

    @testset "allknn" begin
        k = 4
        par_ids, par_dists = allknn(pex, ctx, k)
        gold_ids, gold_dists = allknn(gold, gctx, k)
        @test size(par_ids) == (k, n)
        @test macrorecall(gold_ids, par_ids) == 1.0
    end

    @testset "ctx.scheduler = :sequential still works" begin
        sctx = GenericContext(; scheduler=:sequential)
        k = 3
        for j in 1:length(queries)
            r = knnqueue(KnnSorted, k)
            g = knnqueue(KnnSorted, k)
            search(pex, sctx, queries[j], r)
            search(gold, gctx, queries[j], g)
            @test Set(IdView(r)) == Set(IdView(g))
        end
    end
end
