# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test, Random
Random.seed!(0)

@isdefined(FAST_TESTS) || (const FAST_TESTS = get(ENV, "FAST_TESTS", "false") == "true")

@testset "SpatialAccessTree" begin
    n, m, k, dim, minleaf, dist = (FAST_TESTS ? 10^3 : 10^4), 10^2, 10, 4, 4, Dist.L2()
    db = MatrixDatabase(rand(Float32, dim, n))
    warmqueries = MatrixDatabase(rand(Float32, dim, 2))
    queries = MatrixDatabase(rand(Float32, dim, m))
    seq = ExhaustiveSearch(dist, db)
    ctx = GenericContext()
    Igold, _ = searchbatch(seq, ctx, warmqueries, k)
    Igold, _ = searchbatch(seq, ctx, queries, k)

    combos = [
        (ipart=SatInitialPartition(), sortsat=ProximalSortSat(), recall=0.9),
        (ipart=SatInitialPartition(), sortsat=DistalSortSat(), recall=0.9),
        (ipart=SatInitialPartition(), sortsat=RandomSortSat(), recall=0.9),
        (ipart=RandomInitialPartition(nparts=64, shuffle=false), sortsat=RandomSortSat(), recall=0.9),
        (ipart=RandomInitialPartition(nparts=64, shuffle=true), sortsat=RandomSortSat(), recall=0.9),
    ]
    # FAST_TESTS keeps one combo per initial-partition kind (still exercises every code
    # path) instead of all 5 -- the 3 SatInitialPartition/sortsat combos mostly differ in
    # element order, not in which functions run.
    for e in (FAST_TESTS ? combos[[1, 4, 5]] : combos)
        sortsat, ipart = e.sortsat, e.ipart
        sat = Sat(db; dist)
        index!(sat, ctx, ipart; sortsat, minleaf)

        # checking that the database size and the number of inserted elements is consistent
        @test n == 1 + sum(length(C) for C in sat.children if C !== nothing)
        # checking that cov is consistent with children
        @test all(c -> c >= 0, sat.cov)

        Isat, _ = searchbatch(sat, ctx, warmqueries, k)
        Isat, _ = searchbatch(sat, ctx, queries, k)
        recall = macrorecall(Igold, Isat)
        @test recall >= e.recall

        psat = permutesat(sat)
        @test isperm(psat.π)
        Ip, _ = searchbatch(psat, ctx, warmqueries, k)
        Ip, _ = searchbatch(psat, ctx, queries, k)
        precall = macrorecall(Igold, Ip)
        @test precall >= e.recall
    end
end
