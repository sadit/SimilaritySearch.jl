# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test, Random
Random.seed!(0)

# every object's center is its nearest one (the Dirichlet domain the filters rely on), and its
# stored distance is the distance to that center
function anni_check_partition(anni)
    dist, db = distance(anni), database(anni)
    ok = true
    for u in 1:length(db)
        a = anni.center[u]
        if a == 0
            ok &= u in anni.centers && anni.dcenter[u] == 0
            continue
        end
        d = [Dist.evaluate(dist, db[u], db[c]) for c in anni.centers]
        ok &= isapprox(anni.dcenter[u], d[a]; rtol=1e-5) && d[a] <= minimum(d) * (1 + 1e-5)
    end
    ok
end

function nanni_check_cells(nanni, anni)
    m = length(nanni.centers)
    ok = nanni.centers == anni.centers && length(nanni.cellstart) == m + 1
    ok &= sort(vcat(nanni.members, nanni.centers)) == collect(1:length(database(nanni)))
    for a in 1:m
        lo, hi = nanni.cellstart[a], nanni.cellstart[a+1] - 1
        ok &= issorted(view(nanni.mdist, lo:hi))
        ok &= all(anni.center[u] == a for u in view(nanni.members, lo:hi))
        ok &= hi < lo ? nanni.cov[a] == 0 : nanni.cov[a] == maximum(view(nanni.mdist, lo:hi))
    end
    ok
end

# exact: the same distances as the exhaustive search, rank by rank (ties may swap ids)
same_dists(D, Dgold) = all(isapprox.(D, Dgold; rtol=1e-4, atol=1e-6))

@testset "ANNI and NANNI" begin
    @testset "vectors, L2 (k=$k)" for k in (1, 10)
        n, m = 5000, 40
        db = MatrixDatabase(rand(Float32, 4, n))
        queries = MatrixDatabase(rand(Float32, 4, m))
        ctx = GenericContext(; reporters=[])
        seq = ExhaustiveSearch(Dist.L2(), db)
        _, Dgold = searchbatch(seq, ctx, queries, k)

        anni = ANNI(Dist.L2(), db)
        index!(anni, ctx; k, rng=Xoshiro(1))
        @test 1 < length(anni.centers) < n
        @test anni_check_partition(anni)
        _, D = searchbatch(anni, ctx, queries, k)
        @test same_dists(D, Dgold)

        nanni = NANNI(Dist.L2(), db)
        index!(nanni, ctx; k, rng=Xoshiro(1))
        @test 1 < length(nanni.centers) < n
        _, D = searchbatch(nanni, ctx, queries, k)
        @test same_dists(D, Dgold)

        # the cells of a NANNI are the partition of the ANNI it was built from
        anni2 = ANNI(Dist.L2(), db)
        index!(anni2, ctx; k, rng=Xoshiro(2), step=50, alphastop=Inf)   # one step: 51 centers
        @test length(anni2.centers) == 51
        @test anni_check_partition(anni2)
        nanni2 = SimilaritySearch.Anni._fromanni!(NANNI(Dist.L2(), db), anni2)
        @test nanni_check_cells(nanni2, anni2)
        _, D = searchbatch(nanni2, ctx, queries, k)
        @test same_dists(D, Dgold)

        # low dimension: both prune, NANNI more than ANNI
        ctxa = GenericContext(; reporters=[]); searchbatch(anni, ctxa, queries, k)
        ctxn = GenericContext(; reporters=[]); searchbatch(nanni, ctxn, queries, k)
        ea, en = sum(ctxa.costdists) / m, sum(ctxn.costdists) / m
        @test ea < n / 4
        @test en < ea
    end

    @testset "given training queries" begin
        db = MatrixDatabase(rand(Float32, 3, 3000))
        train = MatrixDatabase(rand(Float32, 3, 64))
        queries = MatrixDatabase(rand(Float32, 3, 30))
        ctx = GenericContext(; reporters=[])
        _, Dgold = searchbatch(ExhaustiveSearch(Dist.L2(), db), ctx, queries, 5)
        for T in (ANNI, NANNI)
            idx = T(Dist.L2(), db)
            index!(idx, ctx; queries=train, k=5, rng=Xoshiro(3))
            _, D = searchbatch(idx, ctx, queries, 5)
            @test same_dists(D, Dgold)
        end
    end

    @testset "permuted copies (cell order)" begin
        db = MatrixDatabase(rand(Float32, 4, 3000))
        queries = MatrixDatabase(rand(Float32, 4, 30))
        ctx = GenericContext(; reporters=[])
        Igold, Dgold = searchbatch(ExhaustiveSearch(Dist.L2(), db), ctx, queries, 7)
        anni = ANNI(Dist.L2(), db); index!(anni, ctx; k=7, rng=Xoshiro(5))
        nanni = NANNI(Dist.L2(), db); index!(nanni, ctx; k=7, rng=Xoshiro(5))
        for (idx, P) in ((anni, permuteanni(anni)), (nanni, permutenanni(nanni)))
            @test isperm(P.π)
            I, D = searchbatch(P, ctx, queries, 7)
            @test same_dists(D, Dgold)
            # the answers carry the original identifiers
            @test all(isapprox(Dist.evaluate(Dist.L2(), queries[j], db[I[i, j]]), D[i, j]; rtol=1e-5) for i in 1:7, j in 1:30)
        end
        P = permuteanni(anni)
        @test P.index.centers == 1:length(anni.centers)
        @test anni_check_partition(P.index)
        P = permutenanni(nanni)
        @test P.index.members == length(nanni.centers)+1:length(db)
    end

    @testset "strings, Levenshtein" begin
        alphabet = collect("abcdefgh")
        mkword() = collect(String(rand(alphabet, rand(3:9))))
        n, m, k = 2000, 30, 5
        dist = Dist.Seqs.Levenshtein()
        db = VectorDatabase([mkword() for _ in 1:n])
        queries = VectorDatabase([mkword() for _ in 1:m])
        ctx = GenericContext(; reporters=[])
        _, Dgold = searchbatch(ExhaustiveSearch(dist, db), ctx, queries, k)
        for T in (ANNI, NANNI)
            idx = T(dist, db)
            index!(idx, ctx; k, rng=Xoshiro(4))
            _, D = searchbatch(idx, ctx, queries, k)
            @test D == Dgold
            P = T === ANNI ? permuteanni(idx) : permutenanni(idx)
            _, D = searchbatch(P, ctx, queries, k)
            @test D == Dgold
        end
    end

    @testset "edge cases" begin
        ctx = GenericContext(; reporters=[])
        @test_throws ArgumentError ANNI(Dist.SqL2(), MatrixDatabase(rand(Float32, 2, 10)))
        @test_throws ArgumentError NANNI(Dist.SqL2(), MatrixDatabase(rand(Float32, 2, 10)))
        for T in (ANNI, NANNI)
            idx = T(Dist.L2(), MatrixDatabase(rand(Float32, 2, 0)))
            index!(idx, ctx)
            @test length(search(idx, ctx, rand(Float32, 2), knnqueue(KnnSorted, 3))) == 0

            one = MatrixDatabase(rand(Float32, 2, 1))
            idx = T(Dist.L2(), one)
            index!(idx, ctx)
            res = search(idx, ctx, rand(Float32, 2), knnqueue(KnnSorted, 3))
            @test length(res) == 1

            idx = T(Dist.L2(), MatrixDatabase(rand(Float32, 2, 100)))
            index!(idx, ctx)
            @test_throws ArgumentError index!(idx, ctx)
        end
    end
end
