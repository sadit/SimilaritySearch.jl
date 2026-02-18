# This file is a part of SimilaritySearch.jl

using SimilaritySearch
const Dist = SimilaritySearch.Dist
using LinearAlgebra, Test

function test_seq(db, queries, dist::Dist.SemiMetric, ksearch, valid_lower=1e-3)
    seq = ExhaustiveSearch(dist, db)
    ctx = getcontext(seq)
    knns = zeros(IdWeight, ksearch, length(queries))
    @time "$(typeof(dist))" knns = searchbatch!(seq, ctx, queries, knns)
    fill!(knns, zero(IdWeight))
    @time "$(typeof(dist))" knns = searchbatch!(seq, ctx, queries, knns)
    #@test_call target_modules=(@__MODULE__,) searchbatch(seq, ctx, queries, ksearch)

    for c in eachcol(knns)
        @test c[1].weight < valid_lower
    end    

end

@testset "Searching vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 4
    db = MatrixDatabase(rand(Float32, 4, 10_000))
    queries = rand(db, 100)
    @info typeof(db), typeof(queries)
    for (recall_lower_bound, dist) in [
        (1.0, Dist.L2()), # 1.0 -> metric, < 1.0 if dist is not a metric
        (1.0, Dist.L1()),
        (1.0, Dist.LInfty()),
        (1.0, Dist.SqL2()),
        (1.0, Dist.Lp(3.0)),
        (0.1, Dist.Lp(0.5)),
        (1.0, Dist.Angle()),
        (1.0, Dist.Cosine())
    ]
        test_seq(db, queries, dist, ksearch)
    end
end

@testset "Searching sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 4
    db = VectorDatabase([create_sequence(5, false) for i in 1:10_000])
    queries = rand(db, 100)
    @info typeof(db), typeof(queries)
    
    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        Dist.Seqs.CommonPrefix(),
        Dist.Seqs.Levenshtein(),
        Dist.Seqs.LCS(),
        Dist.Seqs.StringHamming()
    ]
        test_seq(db, queries, dist, ksearch)
    end
end

@testset "Searching on sets (ordered lists)" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 4
    σ = 10
    db = VectorDatabase([create_sequence(5, true, 1:σ) for i in 1:10_000])
    queries = rand(db, 100)
    @info typeof(db), typeof(queries)

    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        Dist.Sets.Jaccard(),
        Dist.Sets.Dice(),
        Dist.Sets.Intersection(),
        Dist.Sets.RogersTanimoto(σ)
    ]
        test_seq(db, queries, dist, ksearch)
    end
end

@testset "Searching with angle-based distances" begin
    # cosine and angle distance
    ksearch = 4
    X = MatrixDatabase(rand(Float32, 4, 1000))
    queries = rand(X, 100)
    for c in X normalize!(c) end

    test_seq(X, queries, Dist.NormAngle(), ksearch)
    test_seq(X, queries, Dist.NormCosine(), ksearch)
end

@testset "Binary distances" begin
    ksearch = 4
    db = MatrixDatabase(rand(UInt64, 8, 1000))
    queries = rand(db, 100)
    test_seq(db, queries, Dist.Bits.Hamming(), ksearch)
    test_seq(db, queries, Dist.Bits.RogersTanimoto(), ksearch)
    # test_seq(db, queries, BinaryRussellRaoDissimilarity(), ksearch)
end

