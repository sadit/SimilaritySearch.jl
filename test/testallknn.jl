# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch

@testset "allknn" begin
    k = 5
    dist = L2Distance()
    n = 100
    X = MatrixDatabase(rand(Float32, 4, n))
    G = SearchGraph(; db=X, dist)
    @show G.len, G.len[], length(G)
    index!(G)
    @test length(G) == n
    optimize!(G, MinRecall(0.95))
    @test length(G) == n
    gI, gD = allknn(G, k)
    @test size(gI) == size(gD) == (k, n)

    E = ExhaustiveSearch(; db=X, dist)
    eI, eD = allknn(E, k)
    @test size(eI) == size(eD) == (k, n)

    P = ParallelExhaustiveSearch(; db=X, dist)
    pI, pD = allknn(P, k)
    @test size(pI) == size(pD) == (k, n)
    @test macrorecall(eI, gI) > 0.8
    @test macrorecall(eI, pI) > 0.99
    
    @test_call allknn(G, k)
    @test_call allknn(E, k)
    @test_call allknn(P, k)
end

