# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch, LinearAlgebra

@testset "neardup" begin
    dist = CosineDistance()
    X = rand(Float32, 4, 1000)
    db = VectorDatabase(Vector{Float32}[])
    ϵ = 0.1
    D = neardup(SearchGraph(; db, dist), MatrixDatabase(X), ϵ)
    @test all(x -> x <= ϵ, D.dist)
    @test sum(D.dist) > 0
    @test D.map == sort(unique(D.nn))
    
    @test_call neardup(SearchGraph(; db, dist), MatrixDatabase(X), ϵ)
end
