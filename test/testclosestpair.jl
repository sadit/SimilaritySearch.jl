# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, LinearAlgebra

@testset "closestpair" begin
    dist = SimilaritySearch.Dist.Cosine()
    dim, mindist = 2, 1e-4
    db = MatrixDatabase(rand(Float32, dim, 1000))
    G = SearchGraph(; db, dist)
    ctx = SearchGraphContext()
    tG = @elapsed index!(G, ctx)
    tG += @elapsed i, j, d = closestpair(G, ctx)
    @test i != j
    @test d < mindist
    @show i, j, d
    i, j, d = closestpair(G, ctx)
    @test i != j
    @test d < mindist
    @show i, j, d, :parallel
    seq = ExhaustiveSearch(; dist, db)
    ctxseq = getcontext(seq)
    tE = @elapsed i, j, d = closestpair(seq, ctxseq)
    @info "NOTE: the exact method will be faster on small datasets due to the preprocessing step of the approximation method"
    @info "closestpair computation time", :approx => tG, :exact => tE

    # @test_call closestpair(G, ctx; minbatch=-1)
end
