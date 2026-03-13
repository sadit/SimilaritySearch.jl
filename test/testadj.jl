# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra
using SimilaritySearch:
    AdjList32, AdjDict32, StaticAdjList, neighbors_length, packed_neighbors, unpack_edge, add!

@testset "AdjList32" begin
    function radj()
        n = rand([3, 7, 11])
        rand(UInt32(1):UInt32(100), n) |> unique |> sort!
    end

    T = Vector{UInt32}[]
    A = AdjList32(10)
    for i in 1:10
        push!(T, radj())
        add!(A, i, T[end]; linkrev=false)
    end

    let B = StaticAdjList(A)
        #@show collect(A) collect(B)
        @test length(A) == length(B)
        @test neighbors_length.(Ref(A), eachindex(A)) == neighbors_length.(Ref(B), eachindex(B))
        @test packed_neighbors.(Ref(A), eachindex(A)) == packed_neighbors.(Ref(B), eachindex(B))
        for (i, (_A, _B)) in enumerate(zip(packed_neighbors.(Ref(A), eachindex(A)), T))  # only because linkrev=false
            @assert _A == _B "ERROR $i $_A != $_B"
        end
    end

    let B = AdjList32(length(A))
        add!(B, A)
        @test length(A) == length(B)
        @test packed_neighbors.(Ref(A), eachindex(A)) == packed_neighbors.(Ref(B), eachindex(B))
    end

    let B = AdjDict32(length(A))
        add!(B, A)
        @test length(A) == length(B)
        for i in eachindex(B)
            #@info :test i
            @test packed_neighbors(A, i) == packed_neighbors(B, i)
        end
   end
end
