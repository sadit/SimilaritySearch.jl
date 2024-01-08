# This file is a part of SimilaritySearch.jl

using Test, JET, SimilaritySearch, LinearAlgebra

@testset "closestpair" begin
    dist = CosineDistance()
    dim, mindist = 2, 1e-4
    db = MatrixDatabase(rand(Float32, dim, 1000))
    G = SearchGraph(; db, dist)
    ctx = getcontext(G)
    tG = @elapsed index!(G, ctx)
    tG += @elapsed i, j, d = closestpair(G, ctx; minbatch=-1)
    @test i != j
    @test d < mindist
    @show i, j, d
    i, j, d = closestpair(G, ctx)
    @test i != j
    @test d < mindist
    @show i, j, d, :parallel
    tE = @elapsed i, j, d = closestpair(ExhaustiveSearch(; dist, db), ctx)
    @info "NOTE: the exact method will be faster on small datasets due to the preprocessing step of the approximation method"
    @info "closestpair computation time", :approx => tG, :exact => tE
    
    @test_call closestpair(G, ctx; minbatch=-1)
end
