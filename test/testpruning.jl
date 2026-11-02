# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using SimilaritySearch: _prunedcount, _MaskedQueue
using Test, Random
Random.seed!(0)

@testset "PruningSearch: the stretched triangle inequality" begin
    dim, n, nq, k = 16, 5_000, 60, 10
    dist = Dist.L2()
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, nq))
    ctx = GenericContext(; reporters=[])
    gold, _ = searchbatch(ExhaustiveSearch(dist, db), ctx, queries, k)

    @test_throws ArgumentError PruningSearch(index!(ANNI(dist, db), ctx); factor=0)
    @test_throws ArgumentError PruningSearch(index!(ANNI(dist, db), ctx); factor=1.5)

    indexes = [
        "ANNI" => index!(ANNI(dist, db), ctx; rng=Xoshiro(1)),
        "NANNI" => index!(NANNI(dist, db), ctx; rng=Xoshiro(2)),
        "EPTable" => index!(EPTable(dist, db; groups=2), ctx; rng=Xoshiro(3)),
        "NEP" => index!(NEP(dist, db; nfar=1), ctx; rng=Xoshiro(4)),
    ]

    # the mean evaluations and the answers of the stretched search with `factor`
    function measure(idx, factor)
        ids = zeros(UInt32, k, nq); evals = 0
        for i in 1:nq
            res = knnqueue(KnnSorted, k)
            evals += _prunedcount(idx, queries[i], res, Float32(factor))
            ids[:, i] .= sort!(collect(IdView(res)))
        end
        evals / nq, ids
    end

    @testset "$name" for (name, idx) in indexes
        # factor 1 is the exact search: same answers, same evaluations
        exact, exids = measure(idx, 1)
        P = PruningSearch(idx)
        ids, _ = searchbatch(P, ctx, queries, k)
        @test macrorecall(gold, ids) == 1
        @test macrorecall(gold, exids) == 1
        c = knnqueue(KnnSorted, k)
        e1 = (name in ("ANNI", "NANNI") ? SimilaritySearch.Anni._searchcount : SimilaritySearch.ExtremePivots._searchcount)(idx, queries[1], c)
        @test e1 == _prunedcount(idx, queries[1], knnqueue(KnnSorted, k), 1f0)

        # a lower factor evaluates no more, and fewer at 0.3 in dimension 16
        prev = exact
        for f in (0.7, 0.5, 0.3, 0.1)
            ev, _ = measure(idx, f)
            @test ev <= prev
            prev = ev
        end
        @test prev < exact

        # tuned for recall 0.9 on internal queries (masked) by its evaluations
        P = PruningSearch(idx)
        optimize_index!(P, ctx, MinRecall(0.9); ksearch=k, numqueries=64)
        f = P.config[].factor
        @test 0 < f <= 1
        ids, _ = searchbatch(P, ctx, queries, k)
        ev, _ = measure(idx, f)
        @info "PruningSearch over $name: factor $(round(f, digits=3)), recall $(round(macrorecall(gold, ids), digits=3)), $(round(ev, digits=1)) evaluations against $(round(exact, digits=1)) exact"
        @test macrorecall(gold, ids) >= 0.75
        f < 1 && @test ev < exact
    end

    @testset "an internal tuning query does not answer itself" begin
        _, nanni = indexes[2]
        res = knnqueue(KnnSorted, k)
        _prunedcount(nanni, db[7], _MaskedQueue(res, UInt32[7]), 1f0)
        @test !(UInt32(7) in IdView(res))
        @test length(res) == k
    end

    @testset "permuted: tuned inside, wrapped back" begin
        _, nanni = indexes[2]
        Pm = permutenanni(nanni)
        W = PermutedSearchIndex(; index=PruningSearch(Pm.index), Pm.π, Pm.π′)
        ids, _ = searchbatch(W, ctx, queries, k)
        @test macrorecall(gold, ids) == 1
    end
end
