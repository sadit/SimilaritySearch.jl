# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra
using SimilaritySearch:
    AdjList, AdjDict, StaticAdjList, neighbors, add!

@testset "AdjList" begin
    function radj()
        n = rand([3, 7, 11])
        L = unique(rand(UInt32(1):UInt32(100), n))
        sort!(L)
        L
    end

    A = AdjList([radj() for i in 1:10])
    B = StaticAdjList(A)
    let
        #@show collect(A) collect(B)
        @test length(A) == length(B)
        @test [length(neighbors(A, i)) for i in eachindex(A)] == [length(neighbors(B, i)) for i in eachindex(B)]
        @test [neighbors(A, i) for i in eachindex(A)] == [neighbors(B, i) for i in eachindex(B)]
        @test collect(A) == collect(B)
    end

    let C = AdjList(UInt32)
        add!(C, B)
        @test length(A) == length(C)

        for i in eachindex(C)
            @test neighbors(A, i) == neighbors(C, i)
        end

        @test collect(A) == collect(C)
    end

    let C = AdjDict(UInt32)
        add!(C, B)
        @test length(A) == length(C)

        for i in eachindex(C)
            @test neighbors(A, i) == neighbors(C, i)
        end

        @test collect(A) == sort(collect(C), by=first)
    end
end
