# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using LinearAlgebra, Test

function test_exact(db, queries, dist::Dist.Metric, ksearch::Integer, minrecall::AbstractFloat)
    seq = Exact.ExhaustiveSearch(dist, db)
    idx = Exact.BasketList(dist, db, 32)
    ctx = GenericContext()
    gold_knns, _ = searchbatch(seq, ctx, queries, ksearch)
    knns, _ = searchbatch(idx, ctx, queries, ksearch)
    recall = macrorecall(gold_knns, knns)
    @assert recall >= minrecall  "$recall < $minrecall" # just to allow collisions (we are testing with low dimensional and discrete data)
end

@testset "Searching vectors" begin
    ksearch = 4
    dim = 4
    db = MatrixDatabase(rand(Float32, dim, 2_000))
    queries = MatrixDatabase(rand(Float32, dim, 30))
    @info typeof(db), typeof(queries)
    for dist in [
        Dist.L2(),
        Dist.L1(),
        Dist.LInfty(),
        Dist.Angle(),
    ]
        test_exact(db, queries, dist, ksearch, 0.99)
    end
end

@testset "Binary distances" begin
    ksearch = 4
    db = MatrixDatabase(rand(UInt64, 8, 2_000))
    queries = MatrixDatabase(rand(UInt64, 8, 30))

    test_exact(db, queries, Dist.Bits.Hamming(), ksearch, 0.9)
    test_exact(db, queries, Dist.Bits.RogersTanimoto(), ksearch, 0.9)
end
