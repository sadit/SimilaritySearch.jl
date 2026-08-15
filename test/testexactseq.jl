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
    knns_ids = zeros(UInt32, ksearch, length(queries))
    knns_dists = fill(typemax(Float32), ksearch, length(queries))
    @time "$(typeof(dist))" searchbatch!(idx, ctx, queries, knns_ids, knns_dists)
    fill!(knns_ids, zero(UInt32))
    fill!(knns_dists, typemax(Float32))
    @time "$(typeof(dist))" searchbatch!(idx, ctx, queries, knns_ids, knns_dists)
    #@test_call target_modules=(@__MODULE__,) searchbatch(idx, ctx, queries, ksearch)

    for c in eachcol(knns_dists)
        @test c[1] < valid_lower
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

@testset "RadiusSorted/RadiusHeap via ExhaustiveSearch" begin
    dist = Dist.SqL2()
    n, m = 200, 10
    db = MatrixDatabase(rand(Float32, 4, n))
    queries = MatrixDatabase(rand(Float32, 4, m))
    idx = ExhaustiveSearch(dist, db)
    ctx = GenericContext()

    alldists = [Dist.evaluate(dist, queries[j], db[i]) for i in 1:n, j in 1:m]
    sorted_dists = sort(vec(alldists))
    radius = sorted_dists[cld(length(sorted_dists), 5)]  # ~20th percentile, no Statistics dep needed

    for QueueType in (RadiusSorted, RadiusHeap)
        knns = [QueueType(radius) for _ in 1:m]
        searchbatch!(idx, ctx, queries, knns)

        for j in 1:m
            gold = Set(i for i in 1:n if alldists[i, j] <= radius)
            got = Set(x.id for x in IdDistView(knns[j]))
            @test got == gold
        end
    end
end
