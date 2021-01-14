# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Distances
using LinearAlgebra
using Test

function test_vectors(create_index, dist::PreMetric, ksearch, nick; repeat=1, aggregation=:mean)
    @testset "indexing vectors with $nick and $dist" begin
        n = 300 # number of items in the dataset
        m = 30  # number of queries
        dim = 3  # vector's dimension

        db = [rand(Float32, dim) for i in 1:n]
        queries = [rand(Float32, dim) for i in 1:m]

        index = create_index(db)
        @test length(index.db) == n
        push!(index, dist, rand(Float32, dim))
        @test length(index.db) == 1 + n
        perf = Performance(dist, index.db, queries, expected_k=10)
        p = probe(perf, index, dist, repeat=repeat, aggregation=aggregation)
        @show dist, p
        return p
    end
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10

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
        @show recall_lower_bound, dist, dist isa PreMetric
        p = test_vectors((db) -> fit(Sequential, db), dist, ksearch, "Sequential")
        @test p.recall >= recall_lower_bound * 0.99 # to support "numerical" variations
    end
end

function test_sequences(create_index, dist, ksearch)
    n = 300 # number of items in the dataset
    m = 30  # number of queries
    dim = 5  # the length of sequences
    V = collect(1:10)  # vocabulary of the sequences

    function create_item()
        s = rand(V, dim)
        if dist isa Union{JaccardDistance, DiceDistance, IntersectionDissimilarity}
            sort!(s)
            s = unique(s)
        end

        return s
    end
    db = [create_item() for i in 1:n]
    queries = [create_item() for i in 1:m]

    @info "inserting items into the index"
    index = create_index(db)
    @test length(index.db) == n
    perf = Performance(dist, index.db, queries, expected_k=10)
    p = probe(perf, index, dist)
    return p
end

@testset "indexing sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10

    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for (recall_lower_bound, dist) in [
        (1.0, JaccardDistance()),
        (0.1, DiceDistance()),
        (0.1, IntersectionDissimilarity()),
        (0.1, CommonPrefixDissimilarity()),
        (1.0, LevenshteinDistance()),
        (1.0, LcsDistance()),
        (1.0, StringHammingDistance())
    ]
        @testset "indexing sequences with Sequential and $dist" begin
            p = test_sequences((db) -> fit(Sequential, db), dist, ksearch)
            @test p.recall >= recall_lower_bound * 0.9  # "numerical" and discrete factors
        end
        @testset "indexing sequences with PivotTable and $dist" begin
            p = test_sequences((db) -> fit(PivotTable, dist, db, 1), dist, ksearch)
            @test p.recall >= recall_lower_bound * 0.9
        end
    end
end


function test_cos(create_index, dist::PreMetric, ksearch; repeat=1, aggregation=:mean)
    n = 300 # number of items in the dataset
    m = 30  # number of queries
    dim = 3  # vector's dimension

    db = rand(dim, n) |> normalize!
    queries = rand(dim, m) |> normalize!
    #db = [DenseCosine(rand(Float32, dim)) for i in 1:n]
    #queries = [DenseCosine(rand(Float32, dim)) for i in 1:m]

    index = create_index([@view db[:, i] for i in 1:size(db, 2)])
    # optimize!(index, recall=0.9, use_distances=true)

    # push!(index, dist, rand(Float64, dim) |> normalize!) ## we cannot mix @view's and normal vectors
    # @test length(index.db) == 1 + n
    perf = Performance(dist, index.db, [@view queries[:, i] for i in 1:size(queries, 2)], expected_k=ksearch)
    p = probe(perf, index, dist, repeat=repeat, aggregation=aggregation)
    @show p
    return p
end

function test_binhamming(create_index, dist::PreMetric, ksearch, create)
    n = 300 # number of items in the dataset
    m = 30  # number of queries

    db = [create() for i in 1:n]
    queries = [create() for i in 1:m]

    index = create_index(db)
    @test length(index.db) == n
    push!(index, dist, create())
    @test length(index.db) == 1 + n
    perf = Performance(dist, index.db, queries, expected_k=ksearch)
    p = probe(perf, index, dist)
    @show p
    return p
end

@testset "Cosine and Angle distance" begin
    # cosine and angle distance
    ksearch = 10
    @testset "indexing vectors with Sequential with cos or angle's distance" begin
        p1 = test_cos((db) -> fit(Sequential, db), AngleDistance(), ksearch, repeat=3, aggregation=:median)
        p2 = test_cos((db) -> fit(Sequential, db), AngleDistance(), ksearch, repeat=3, aggregation=:min)
        p3 = test_cos((db) -> fit(Sequential, db), CosineDistance(), ksearch, repeat=3, aggregation=:max)
        p4 = test_cos((db) -> fit(Sequential, db), CosineDistance(), ksearch, repeat=3, aggregation=:mean)
        @show p1, p2, p3, p4
        @test p1.recall > 0.99
        @test p2.recall > 0.99
        @test p3.recall > 0.99
        @test p4.recall > 0.99
    end
end

@testset "Hamming distance" begin
    # cosine and angle distance
    ksearch = 10
    dist = StringHammingDistance()
    @testset "indexing vectors with Sequential with StringHammingDistance" begin
        p = test_binhamming((db) -> sss(dist, db, 0.35), dist, ksearch, () -> UInt32[rand(UInt32) for i in 1:7])
        @test p.recall > 0.9

        p = test_binhamming((db) -> distant_tournament(dist, db, 16, 3), dist, ksearch, () -> rand(UInt64))
        @test p.recall > 0.9
    end
end