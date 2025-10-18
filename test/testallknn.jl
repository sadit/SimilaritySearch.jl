# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch


@testset "allknn" begin
    k = 5
    dist = L2Distance()
    n = 100
    X = MatrixDatabase(rand(Float32, 4, n))

    E = ExhaustiveSearch(; db=X, dist)
    ectx = getcontext(E)

    @time "ExhaustiveSearch allknn" gold_knns = allknn(E, ectx, k)
    #@test_call target_modules=(@__MODULE__,) allknn(E, ectx, k)
    @test size(gold_knns) == (k, n)

    P = ParallelExhaustiveSearch(; db=X, dist)
    @time "ParallelExhaustiveSearch allknn" par_knns = allknn(P, ectx, k)
    #@test_call target_modules=(@__MODULE__,) allknn(P, ectx, k)
    @test size(par_knns) == (k, n)
    @test macrorecall(gold_knns, par_knns) > 0.99

    #=G = SearchGraph(; db=X, dist)
    ctx = getcontext(G)
    @show G.len, G.len[], length(G)
    index!(G, ctx)
    @test length(G) == n
    optimize_index!(G, ctx, MinRecall(0.95))
    @test length(G) == n
    @time "SearchGraph allknn" knns = allknn(G, ctx, k)
    @test size(knns) == (k, n)
    @test_call target_modules=(@__MODULE__,) allknn(G, ctx, k)
    @test macrorecall(gold_knns, knns) > 0.8=#
end

