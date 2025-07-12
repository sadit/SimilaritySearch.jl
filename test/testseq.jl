# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Distances, LinearAlgebra
using Test, JET

function test_seq(db, queries, dist::SemiMetric, ksearch, valid_lower=1e-3)
    seq = ExhaustiveSearch(dist, db)
    ctx = getcontext(seq)
    knns = zeros(IdWeight, ksearch, length(queries))
    @time knns = searchbatch!(seq, ctx, queries, knns)
    fill!(knns, zero(IdWeight))
    @time knns = searchbatch!(seq, ctx, queries, knns)
    #@test_call target_modules=(@__MODULE__,) searchbatch(seq, ctx, queries, ksearch)

    for c in eachcol(knns)
        @test c[1].weight < valid_lower
    end    

end

@testset "indexing vectors with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 4
    db = MatrixDatabase(rand(Float32, 4, 100_000))
    queries = rand(db, 1000)
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
    ksearch = 4
    db = VectorDatabase([create_sequence(5, false) for i in 1:100000])
    queries = rand(db, 1000)
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
    ksearch = 4
    db = VectorDatabase([create_sequence(5, true) for i in 1:100000])
    queries = rand(db, 1000)
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
    ksearch = 4
    X = MatrixDatabase(rand(Float32, 4, 1000))
    queries = rand(X, 1000)
    for c in X normalize!(c) end

    test_seq(X, queries, NormalizedAngleDistance(), ksearch)
    test_seq(X, queries, NormalizedCosineDistance(), ksearch)
end

@testset "Binary hamming distance" begin
    ksearch = 4
    db = MatrixDatabase(rand(UInt64, 8, 1000))
    queries = rand(db, 1000)
    test_seq(db, queries, BinaryHammingDistance(), ksearch)
end
