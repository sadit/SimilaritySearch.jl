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

    # concurrent appends (#138): many to the same node, and to nodes that share a stripe, from
    # every thread, mixing add! (glock) and add_resized! (the stripe alone); each list must end
    # up with exactly what was appended to it
    let m = 4000, S = length(SimilaritySearch.ADJ_STRIPES), C = resize!(AdjList(UInt32), S + 8)
        nodes = [1, 2, 1 + S, 8]
        Threads.@threads for t in 1:m
            for n in nodes
                isodd(t) ? add!(C, n, UInt32[t]) : SimilaritySearch.add_resized!(C, n, UInt32[t])
            end
        end
        for n in nodes
            @test sort(neighbors(C, n)) == UInt32.(1:m)
        end
        # add! grows the list under glock while other threads call it
        Threads.@threads for t in 1:m
            add!(C, S + 8 + t, UInt32[t])
        end
        @test length(C) == S + 8 + m
        @test all(neighbors(C, S + 8 + t) == UInt32[t] for t in 1:m)
        # add_resized! beyond the end, from a serial caller, grows as add! does
        SimilaritySearch.add_resized!(C, length(C) + 2, UInt32[7])
        @test neighbors(C, length(C)) == UInt32[7]
    end

    let C = AdjDict(UInt32)
        add!(C, B)
        @test length(A) == length(C)

        for i in eachindex(C)
            @test neighbors(A, i) == neighbors(C, i)
        end

        @test collect(A) == sort(collect(C), by=first)
    end

    let C = AdjDict(String, UInt32)
        add!(C, "alpha", UInt32[1, 5, 10])
        add!(C, "beta", UInt32[2, 3])
        @test length(C) == 2
        @test neighbors(C, "alpha") == UInt32[1, 5, 10]
        @test neighbors(C, "beta") == UInt32[2, 3]
        @test neighbors(C, "gamma") === nothing
    end

    let C = AdjDict(NTuple{4, UInt8}, UInt32)
        k1 = (0x01, 0x02, 0x03, 0x04)
        k2 = (0x0a, 0x0b, 0x0c, 0x0d)
        add!(C, k1, UInt32[42])
        add!(C, k2, UInt32[7, 9])
        @test length(C) == 2
        @test neighbors(C, k1) == UInt32[42]
        @test neighbors(C, k2) == UInt32[7, 9]
    end
end
