# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using LinearAlgebra, Test

function create_sequence(dim, sort, range=1:10)
    s = rand(range, dim)
    if sort
        sort!(s)
        s = unique(s)
    end
    s
end

function test_seq(db, queries, dist::Dist.SemiMetric, ksearch; valid_lower::Float32=1f-3)
    idx = Exact.ExhaustiveSearch(dist, db)
    # idx = Exact.BasketList(dist, db, 256)
    ctx = GenericContext()
    knns = zeros(IdDist, ksearch, length(queries))
    @time "$(typeof(dist))" knns = searchbatch!(idx, ctx, queries, knns)
    fill!(knns, zero(IdDist))
    @time "$(typeof(dist))" knns = searchbatch!(idx, ctx, queries, knns)
    #@test_call target_modules=(@__MODULE__,) searchbatch(idx, ctx, queries, ksearch)

    for c in eachcol(knns)
        @test c[1].dist < valid_lower
    end    

end

@testset "Searching vectors" begin
    ksearch = 4
    db = MatrixDatabase(rand(Float32, 4, 200))
    queries = rand(db, 20)
    for dist in [
        Dist.SqL2(),
        Dist.L1()
    ]
        test_seq(db, queries, dist, ksearch)
    end
end


@testset "Searching sequences" begin
    ksearch = 4
    db = VectorDatabase([create_sequence(5, false) for i in 1:200])
    queries = rand(db, 20)
    
    test_seq(db, queries, Dist.Seqs.Levenshtein(), ksearch)
end


@testset "Searching on sets (ordered lists)" begin
    ksearch = 4
    σ = 10
    db = VectorDatabase([create_sequence(5, true, 1:σ) for i in 1:200])
    queries = rand(db, 20)

    test_seq(db, queries, Dist.Sets.Jaccard(), ksearch)
end

@testset "Searching with angle-based distances" begin
    ksearch = 4
    X = MatrixDatabase(rand(Float32, 4, 200))
    queries = rand(X, 30)
    for c in X normalize!(c) end

    test_seq(X, queries, Dist.NormCosine(), ksearch)
end

@testset "Binary distances" begin
    ksearch = 4
    db = MatrixDatabase(rand(UInt64, 8, 200))
    queries = rand(db, 20)
    test_seq(db, queries, Dist.Bits.Hamming(), ksearch)
end
