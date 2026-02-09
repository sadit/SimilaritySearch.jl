# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, StatsBase, SimilaritySearch.AdjacencyLists


@testset "allknn" begin
    k = 6
    dist = L2Distance()
    n = 100
    db = MatrixDatabase(rand(Float32, 4, n))

    E = ExhaustiveSearch(; db, dist)
    ectx = getcontext(E)

    @time "ExhaustiveSearch allknn" gold_knns = allknn(E, ectx, k)
    #@test_call target_modules=(@__MODULE__,) allknn(E, ectx, k)
    @test size(gold_knns) == (k, n)
    for i in 1:k
        @info "All KNN quartile $i-th:"
        @info i => quantile(collect(DistView(gold_knns[i, :])), 0:0.25:1)
    end

    #= P = ParallelExhaustiveSearch(; db=X, dist)
    @time "ParallelExhaustiveSearch allknn" par_knns = allknn(P, ectx, k)
    @test size(par_knns) == (k, n)
    @test macrorecall(gold_knns, par_knns) > 0.99
    =#

    G = SearchGraph(; db, dist)
    ctx = getcontext(G)
    index!(G, ctx)
    @test length(G) == n
    optimize_index!(G, ctx, MinRecall(0.95))
    @time "SearchGraph allknn" knns = allknn(G, ctx, k)
    @test size(knns) == (k, n)
    recall = macrorecall(gold_knns, knns)
    @show recall recall > 0.8
    @show recall quantile(neighbors_length.(Ref(G.adj), 1:length(G)), 0:0.25:1)
end

