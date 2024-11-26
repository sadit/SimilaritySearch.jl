# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch

@testset "HSP" begin
    k = 32
    dist = L2Distance()
    n = 1000
    X = MatrixDatabase(rand(Float32, 2, n))

    X1 = hsp_queries(dist, X, X, k)

    X1 = sort!(length.(X1))
    @show X1[[1, n ÷ 4, n ÷ 2, round(Int, 0.75 * n), n]]
end


