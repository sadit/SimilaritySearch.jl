# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch

@testset "allknn" begin
    k = 5
    dist = L2Distance()
    X = MatrixDatabase(rand(Float32, 4, 100))
    G = SearchGraph(; db=X, dist)
    index!(G)
    optimize!(G, MinRecall(0.95))
    gI, gD = allknn(G, k)

    E = ExhaustiveSearch(; db=X, dist)
    eI, eD = allknn(E, k)

    P = ParallelExhaustiveSearch(; db=X, dist)
    pI, pD = allknn(P, k)

    @test macrorecall(eI, gI) > 0.8
    @test macrorecall(eI, pI) > 0.99
end