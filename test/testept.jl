# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test, Random
Random.seed!(0)

# in every group, every object sits with the pivot that sees it farthest from its mean, and its
# stored distance is the distance to that pivot
function ept_check_groups(ept)
    dist, db = distance(ept), database(ept)
    ok = allunique(ept.pivots) && count(ept.ispivot) == length(ept.pivots)
    ok &= isperm(ept.order) && issorted(ept.blockstart)
    for b in eachindex(ept.bpiv)   # blocks: one first-group pivot each, sorted by the distance to it
        lo, hi = ept.blockstart[b], ept.blockstart[b+1] - 1
        ok &= all(==(ept.bpiv[b]), view(ept.piv, lo:hi, 1)) && issorted(view(ept.pdist, lo:hi, 1))
    end
    for g in 1:size(ept.piv, 2)
        P = findall(==(g), ept.pgroup)
        ok &= !isempty(P)
        for j in 1:length(db)
            u = ept.order[j]
            p = ept.piv[j, g]
            ok &= ept.pgroup[p] == g
            ok &= isapprox(ept.pdist[j, g], Dist.evaluate(dist, db[u], db[ept.pivots[p]]); rtol=1e-5, atol=1e-6)
            e = abs(ept.pdist[j, g] - ept.pmean[p])
            best = maximum(abs(Dist.evaluate(dist, db[u], db[ept.pivots[i]]) - ept.pmean[i]) for i in P)
            ok &= e >= best * (1 - 1e-4) - 1e-6
        end
    end
    ok
end

same_dists(D, Dgold) = all(isapprox.(D, Dgold; rtol=1e-4, atol=1e-6))

@testset "EPTable" begin
    @testset "vectors, L2 (groups=$G, cost=$cost)" for G in (1, 4), cost in (:model, :distances)
        n, m, k = 4000, 40, 5
        db = MatrixDatabase(rand(Float32, 4, n))
        queries = MatrixDatabase(rand(Float32, 4, m))
        ctx = GenericContext(; reporters=[])
        _, Dgold = searchbatch(ExhaustiveSearch(Dist.L2(), db), ctx, queries, k)
        ept = EPTable(Dist.L2(), db; groups=G)
        index!(ept, ctx; cost, k, rng=Xoshiro(1))
        @test length(ept.pivots) >= G
        @test ept_check_groups(ept)
        cctx = GenericContext(; reporters=[])
        _, D = searchbatch(ept, cctx, queries, k)
        @test same_dists(D, Dgold)
        @test sum(cctx.costdists) / m < n / 2   # low dimension: it prunes
        P = permuteept(ept)
        @test isperm(P.π) && P.index.order == 1:n
        @test ept_check_groups(P.index)
        I, D = searchbatch(P, ctx, queries, k)
        @test same_dists(D, Dgold)
        @test all(isapprox(Dist.evaluate(Dist.L2(), queries[j], db[I[i, j]]), D[i, j]; rtol=1e-5) for i in 1:k, j in 1:m)
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
        for cost in (:model, :distances)
            ept = EPTable(dist, db; groups=3)
            index!(ept, ctx; cost, k, rng=Xoshiro(2))
            _, D = searchbatch(ept, ctx, queries, k)
            @test D == Dgold
            _, D = searchbatch(permuteept(ept), ctx, queries, k)
            @test D == Dgold
        end
    end

    @testset "edge cases" begin
        ctx = GenericContext(; reporters=[])
        @test_throws ArgumentError EPTable(Dist.SqL2(), MatrixDatabase(rand(Float32, 2, 10)))
        @test_throws ArgumentError EPTable(Dist.L2(), MatrixDatabase(rand(Float32, 2, 10)); groups=0)
        ept = EPTable(Dist.L2(), MatrixDatabase(rand(Float32, 2, 0)))
        index!(ept, ctx)
        @test length(search(ept, ctx, rand(Float32, 2), knnqueue(KnnSorted, 3))) == 0
        ept = EPTable(Dist.L2(), MatrixDatabase(rand(Float32, 2, 3)); groups=4)
        @test_throws ArgumentError index!(ept, ctx)   # 3 objects cannot feed 4 groups
        ept = EPTable(Dist.L2(), MatrixDatabase(rand(Float32, 2, 200)); groups=2)
        index!(ept, ctx; maxpivots=5)
        @test length(ept.pivots) <= 10
        res = search(ept, ctx, rand(Float32, 2), knnqueue(KnnSorted, 3))
        @test length(res) == 3
        @test_throws ArgumentError index!(ept, ctx)
        @test_throws ArgumentError index!(EPTable(Dist.L2(), MatrixDatabase(rand(Float32, 2, 50))), ctx; cost=:other)
    end
end

@testset "NEP" begin
    @testset "vectors, L2 (nfar=$F)" for F in (0, 1, 2)
        n, m, k = 4000, 40, 5
        db = MatrixDatabase(rand(Float32, 4, n))
        queries = MatrixDatabase(rand(Float32, 4, m))
        ctx = GenericContext(; reporters=[])
        _, Dgold = searchbatch(ExhaustiveSearch(Dist.L2(), db), ctx, queries, k)
        nep = NEP(Dist.L2(), db; nfar=F)
        index!(nep, ctx; k, rng=Xoshiro(1))
        @test length(nep.pivots) > 1
        # cells: every member in the cell of its nearest pivot, sorted by distance; far pivots by descending d - μ
        ok = sort(vcat(nep.members, nep.pivots)) == collect(1:n)
        for a in eachindex(nep.pivots)
            lo, hi = nep.cellstart[a], nep.cellstart[a+1] - 1
            ok &= issorted(view(nep.mdist, lo:hi))
            for j in lo:hi
                u = nep.members[j]
                d = [Dist.evaluate(Dist.L2(), db[u], db[p]) for p in nep.pivots]
                ok &= isapprox(d[a], minimum(d); rtol=1e-5) && isapprox(nep.mdist[j], d[a]; rtol=1e-5)
                s = d .- nep.pmean
                fs = [s[nep.far[(j-1)*F+t]] for t in 1:F]
                ok &= issorted(fs; rev=true) && (F == 0 || isapprox(fs[1], maximum(s); rtol=1e-4, atol=1e-6))
            end
        end
        @test ok
        cctx = GenericContext(; reporters=[])
        _, D = searchbatch(nep, cctx, queries, k)
        @test same_dists(D, Dgold)
        @test sum(cctx.costdists) / m < n / 2
        I, D = searchbatch(permutenep(nep), ctx, queries, k)
        @test same_dists(D, Dgold)
        @test all(isapprox(Dist.evaluate(Dist.L2(), queries[j], db[I[i, j]]), D[i, j]; rtol=1e-5) for i in 1:k, j in 1:m)
    end

    @testset "strings, Levenshtein" begin
        alphabet = collect("abcdefgh")
        mkword() = collect(String(rand(alphabet, rand(3:9))))
        dist = Dist.Seqs.Levenshtein()
        db = VectorDatabase([mkword() for _ in 1:2000])
        queries = VectorDatabase([mkword() for _ in 1:30])
        ctx = GenericContext(; reporters=[])
        _, Dgold = searchbatch(ExhaustiveSearch(dist, db), ctx, queries, 5)
        nep = NEP(dist, db; nfar=2)
        index!(nep, ctx; k=5, rng=Xoshiro(2))
        _, D = searchbatch(nep, ctx, queries, 5)
        @test D == Dgold
    end

    @test_throws ArgumentError NEP(Dist.SqL2(), MatrixDatabase(rand(Float32, 2, 10)))
    @test_throws ArgumentError NEP(Dist.L2(), MatrixDatabase(rand(Float32, 2, 10)); nfar=-1)
    nep = NEP(Dist.L2(), MatrixDatabase(rand(Float32, 2, 0))); index!(nep, GenericContext(; reporters=[]))
    @test length(search(nep, GenericContext(; reporters=[]), rand(Float32, 2), knnqueue(KnnSorted, 3))) == 0
end
