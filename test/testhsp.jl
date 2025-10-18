# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, StatsBase

@testset "HSP" begin
    k = 32
    dist = L2Distance()
    n = 1000
    db = MatrixDatabase(rand(Float32, 2, n))
    E = ExhaustiveSearch(; dist, db)
    knns = searchbatch(E, getcontext(E), db, k)
    hsp_matrix, hsp_knns = hsp_queries(dist, db, db, knns)
    @show quantile(length.(hsp_knns), 0:0.1:1)
end


