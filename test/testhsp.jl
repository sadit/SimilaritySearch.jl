# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch, StatsBase

@testset "HSP" begin
    k = 32
    dist = L2Distance()
    n = 1000
    X = MatrixDatabase(rand(Float32, 2, n))
    X1 = hsp_queries(dist, X, X, k)
    @show quantile(length.(X1.knns), 0:0.1:1)
end


