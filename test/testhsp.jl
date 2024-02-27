# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch

@testset "HSP" begin
    k = 32
    dist = L2Distance()
    n = 1000
    X = MatrixDatabase(rand(Float32, 2, n))

    for h in hsp_queries(dist, X, X, k)
        @info length(h) h
    end
end


