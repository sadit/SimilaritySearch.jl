# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch, LinearAlgebra
using SimilaritySearch:
    AdjacencyList, StaticAdjacencyList, neighbors

@testset "AdjacencyList" begin
    function radj()
        n = rand([3, 7, 11])
        L = unique(rand(UInt32(1):UInt32(100), n))
        sort!(L)
        L
    end

    A = AdjacencyList([radj() for i in 1:10])
    B = StaticAdjacencyList(A)
    @test length(A) == length(B)
    @test [length(neighbors(A, i)) for i in eachindex(A)] == [length(neighbors(B, i)) for i in eachindex(B)]
    @test [neighbors(A, i) for i in eachindex(A)] == [neighbors(B, i) for i in eachindex(B)]

    C = AdjacencyList(B)
    @test length(A) == length(B)

    for i in eachindex(C)
        @test neighbors(A, i) == neighbors(C, i)
    end

    @test A.end_point == C.end_point
end
