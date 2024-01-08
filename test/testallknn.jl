# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch

@testset "allknn" begin
    k = 5
    dist = L2Distance()
    n = 100
    X = MatrixDatabase(rand(Float32, 4, n))
    G = SearchGraph(; db=X, dist)
    ctx = getcontext(G)
    @show G.len, G.len[], length(G)
    index!(G, ctx)
    @test length(G) == n
    optimize_index!(G, ctx, MinRecall(0.95))
    @test length(G) == n
    gI, gD = allknn(G, ctx, k)
    @test size(gI) == size(gD) == (k, n)

    E = ExhaustiveSearch(; db=X, dist)
    ectx = getcontext(E)
    
    eI, eD = allknn(E, ectx, k)
    @test size(eI) == size(eD) == (k, n)

    P = ParallelExhaustiveSearch(; db=X, dist)
    pI, pD = allknn(P, ectx, k)
    @test size(pI) == size(pD) == (k, n)
    @test macrorecall(eI, gI) > 0.8
    @test macrorecall(eI, pI) > 0.99
    
    @test_call allknn(G, ctx, k)
    @test_call allknn(E, ectx, k)
    @test_call allknn(P, ectx, k)
end

