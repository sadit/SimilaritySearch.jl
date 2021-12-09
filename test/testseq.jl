# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Distances
using LinearAlgebra
using Test

function test_seq(db, queries, dist::PreMetric, ksearch, valid_lower=1e-3)
    seq = ExhaustiveSearch(dist, db)
    reslist = [KnnResult(ksearch) for i in eachindex(queries)]

    t = @timed searchbatch(seq, queries, reslist)
    @info t.gcstats
    for r in reslist
        @test minimum(r) < valid_lower
    end    
end

@testset "indexing vectors with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 3
    db = MatrixDatabase(rand(4, 1000))
    queries = rand(db, 100)
    @info typeof(db), typeof(queries)
    for (recall_lower_bound, dist) in [
        (1.0, L2Distance()), # 1.0 -> metric, < 1.0 if dist is not a metric
        (1.0, L1Distance()),
        (1.0, LInftyDistance()),
        (1.0, SqL2Distance()),
        (1.0, LpDistance(3.0)),
        (0.1, LpDistance(0.5)),
        (1.0, AngleDistance()),
        (1.0, CosineDistance())
    ]
        test_seq(db, queries, dist, ksearch)
    end
end

@testset "indexing sequences with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    db = VectorDatabase([create_sequence(5, false) for i in 1:1000])
    queries = rand(db, 100)
    @info typeof(db), typeof(queries)
    
    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        CommonPrefixDissimilarity(),
        LevenshteinDistance(),
        LcsDistance(),
        StringHammingDistance()
    ]
        test_seq(db, queries, dist, ksearch)
    end
end

@testset "indexing sets with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    db = VectorDatabase([create_sequence(5, true) for i in 1:1000])
    queries = rand(db, 100)
    @info typeof(db), typeof(queries)

    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        JaccardDistance(),
        DiceDistance(),
        IntersectionDissimilarity()
    ]
        test_seq(db, queries, dist, ksearch)
    end
end

@testset "Normalized Cosine and Normalized Angle distances" begin
    # cosine and angle distance
    ksearch = 10
    X = MatrixDatabase(rand(4, 1000))
    queries = rand(X, 100)
    normalize!.(X)

    test_seq(X, queries, NormalizedAngleDistance(), ksearch)
    test_seq(X, queries, NormalizedCosineDistance(), ksearch)
end

@testset "Binary hamming distance" begin
    ksearch = 10
    db = MatrixDatabase(rand(UInt64, 8, 1000))
    queries = rand(db, 100)
    test_seq(db, queries, BinaryHammingDistance(), ksearch)
end
