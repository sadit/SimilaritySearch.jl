# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Distances
using LinearAlgebra
using Test

function test_seq(db, dist::PreMetric, ksearch, valid_lower=1e-3)
    seq = ExhaustiveSearch(dist, db)

    for i in rand(1:length(db), 100)
        res = search(seq, db[i], ksearch)
        @test first(res).dist < valid_lower
    end    
end


@testset "indexing vectors with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 3
    db = create_vectors(1000, 4)

    for (recall_lower_bound, dist) in [
        (1.0, L2Distance()), # 1.0 -> metric, < 1.0 if dist is not a metric
        (1.0, L1Distance()),
        (1.0, LInftyDistance()),
        (0.1, SqL2Distance()),
        (1.0, LpDistance(3.0)),
        (0.1, LpDistance(0.5)),
        (1.0, AngleDistance()),
        (1.0, CosineDistance())
    ]
        test_seq(db, dist, ksearch)
    end
end

@testset "indexing sequences with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    db = [create_sequence(5, false) for i in 1:1000]
    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        CommonPrefixDissimilarity(),
        LevenshteinDistance(),
        LcsDistance(),
        StringHammingDistance()
    ]
        test_seq(db, dist, ksearch)
    end
end

@testset "indexing sets with ExhaustiveSearch" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    db = [create_sequence(5, true) for i in 1:1000]
    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        JaccardDistance(),
        DiceDistance(),
        IntersectionDissimilarity()
    ]
        test_seq(db, dist, ksearch)
    end
end

@testset "Normalized Cosine and Normalized Angle distances" begin
    # cosine and angle distance
    ksearch = 10
    db = create_vectors(1000, 4, true)

    test_seq(db, NormalizedAngleDistance(), ksearch)
    test_seq(db, NormalizedAngleDistance(), ksearch)
    test_seq(db, NormalizedCosineDistance(), ksearch)
    test_seq(db, NormalizedCosineDistance(), ksearch)
end

@testset "Binary hamming distance" begin
    n = 1000 # number of items in the dataset
    ksearch = 10
    db = [rand(UInt32, 8) for i in 1:n]

    test_seq(db, BinaryHammingDistance(), ksearch)
end
