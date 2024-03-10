# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra

@testset "farthest first traversal" begin 
    dist = L2Distance()
    X = rand(Float32, 4, 300)
    res = fft(dist, MatrixDatabase(X), 30)
    @test Set(res.centers) == Set(res.nn)
    @test all(res.dmax .>= res.dists)
end

