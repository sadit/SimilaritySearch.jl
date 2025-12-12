# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, SparseArrays, LinearAlgebra

@testset "test database abstractions" begin
    X = rand(Float32, 4, 100)
    A = MatrixDatabase(X)
    B = VectorDatabase(X)
    C = BlockMatrixDatabase(X)
    D = B[1:100]
    @test D isa SubDatabase
    @test X === A.matrix
    @test X == hcat(B.vecs...)
    @test X[:, 1:100] == C.blocks[1][:, 1:100]
    @test X == hcat(C...)
    @test length.([A, B, C, D]) == [100, 100, 100, 100]
    @test collect(A) == collect(B) == collect(C) == collect(D)
    for i in rand(1:100, 10)
        for v in getindex.([A, B, C, D], i)
            @test v == X[:, i]
        end
    end

    A[1] = 0
    @test sum(A[1]) == 0
    @test A[1] == X[:, 1]
    @test_throws DimensionMismatch A[2] = [1, 2]
    @test_throws DimensionMismatch C[2] = [1, 2]
    A[2] = [1, 2, 3, 4]
    @test A[2] == [1, 2, 3, 4]
    B[1] = [1, 2]
    @test typeof(B[1]) == Vector{Float32}
    @test B[1] == [1, 2]

    A = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZαβγδι"
    F = VectorDatabase(type=String)
    for i in 1:100
        push_item!(F, String(rand(A, 5)))
    end

    @test length(F) == 100
    @test eltype(F) == String

    G = sparse(rand(1:10000, 30), rand(1:10000, 30), rand(30))
    H = MatrixDatabase(G)
    @test all([norm(H[i]) == norm(G[:, i]) for i in rand(1:size(G, 2), 100)])
end
