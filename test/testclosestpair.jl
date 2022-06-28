# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra

@testset "closestpair" begin
    dist = CosineDistance()
    dim, mindist = 2, 1e-4
    db = MatrixDatabase(rand(Float32, dim, 1000))
    G = SearchGraph(; db, dist)
    tG = @elapsed index!(G)
    tG += @elapsed i, j, d = closestpair(G; minbatch=-1)
    @test i != j
    @test d < mindist
    @show i, j, d
    i, j, d = closestpair(G)
    @test i != j
    @test d < mindist
    @show i, j, d, :parallel
    tE = @elapsed i, j, d = closestpair(ExhaustiveSearch(; dist, db))
    @info "NOTE: the exact method will be faster on small datasets due to the preprocessing step of the approximation method"
    @info "closestpair computation time", :approx => tG, :exact => tE
end