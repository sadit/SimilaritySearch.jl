# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra

@testset "farthest first traversal" begin 
    dist = SimilaritySearch.Dist.L2()
    X = rand(Float32, 4, 30)
    k = 10
    res = fft(dist, MatrixDatabase(X), k)
    @test k == length(res.centers)
    @test Set(res.centers) == Set(res.nn)
    @test all(res.dmax .>= res.dists)
end


