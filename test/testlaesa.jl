using SimilaritySearch
using Base.Test

function test_cos(create_index, ksearch, nick; repeat=1, aggregation=:mean)
    @testset "indexing vectors with $nick with cos or angle's distance" begin
        n = 1000 # number of items in the dataset
        m = 100  # number of queries
        dim = 3  # vector's dimension

        db = [DenseCosine(rand(Float32, dim)) for i in 1:n]
        queries = [DenseCosine(rand(Float32, dim)) for i in 1:m]

        index = create_index(db)
        # optimize!(index, recall=0.9, use_distances=true)
        @test length(index.db) == n
        perf = Performance(index.db, index.dist, queries, expected_k=10)
        p = probe(perf, index, use_distances=false, repeat=repeat, aggregation=aggregation)
        @show p
        return p
    end
end

function test_vectors(create_index, dist, ksearch, nick; repeat=1, aggregation=:mean)
    @testset "indexing vectors with $nick and $dist" begin
        n = 1000 # number of items in the dataset
        m = 100  # number of queries
        dim = 3  # vector's dimension

        db = [rand(Float32, dim) for i in 1:n]
        queries = [rand(Float32, dim) for i in 1:m]

        index = create_index(db)
        # optimize!(index, recall=0.9, use_distances=true)
        @test length(index.db) == n
        perf = Performance(index.db, dist, queries, expected_k=10)
        p = probe(perf, index, use_distances=false, repeat=repeat, aggregation=aggregation)
        @show dist, p
        return p
    end
end

function test_sequences(create_index, dist, ksearch, nick)
    @testset "indexing sequences with $nick and $dist" begin
        n = 1000 # number of items in the dataset
        m = 100  # number of queries
        dim = 5  # the length of sequences
        V = collect(1:10)  # vocabulary of the sequences

        function create_item()
            s = rand(V, dim)
            if dist isa JaccardDistance || dist isa DiceDistance || dist isa IntersectionDistance
                sort!(s)
                s = unique(s)
            end

            return s
        end
        db = [create_item() for i in 1:n]
        queries = [create_item() for i in 1:m]

        info("inserting items into the index")
        # index = Laesa(db, dist, σ)
        index = create_index(db)
        # optimize!(index, recall=0.9, use_distances=true)
        # info("done; now testing")
        @test length(index.db) == n
        perf = Performance(index.db, dist, queries, expected_k=10)
        p = probe(perf, index, use_distances=true)
        # @show dist, p
        return p
    end
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    local index

    for (recall_lower_bound, dist) in [
        (1.0, L2Distance()), # 1.0 -> metric, < 1.0 if dist is not a metric
        (1.0, L1Distance()),
        (1.0, LInfDistance()),
        # (1.0, AngleDistance()),
        (1.0, LpDistance(3)),
        (0.1, LpDistance(0.5))
    ]
        @show recall_lower_bound, dist
        p = test_vectors((db) -> Sequential(db, dist), dist, ksearch, "Sequential")
        @test p.recall >= recall_lower_bound * 0.99 # not 1 to allow some "numerical" deviations

        p = test_vectors((db) -> Laesa(db, dist, 16), dist, ksearch, "Laesa")
        @test p.recall >= recall_lower_bound * 0.99 # not 1 to allow some "numerical" deviations
    end

    p1 = test_cos((db) -> Sequential(db, AngleDistance()), ksearch, "Sequential", repeat=3, aggregation=:median)
    p2 = test_cos((db) -> Sequential(db, AngleDistance()), ksearch, "Sequential", repeat=3, aggregation=:min)
    @show p1, p2
    @test p1.recall > 0.999
    @test p2.recall > 0.99
end

@testset "indexing sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    local index

    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for (recall_lower_bound, dist) in [
        (1.0, JaccardDistance()),
        (0.1, DiceDistance()),
        (0.1, IntersectionDistance()),
        (0.1, CommonPrefixDistance()),
        (1.0, LevDistance()),
        (1.0, LcsDistance()),
        (1.0, HammingDistance())
    ]
        p = test_sequences((db) -> Sequential(db, dist), dist, ksearch, "Sequential")
        @test p.recall >= recall_lower_bound * 0.99  # not 1 to allow some "numerical" deviations
        dist.calls = 0
        p = test_sequences((db) -> Laesa(db, dist, 1), dist, ksearch, "Laesa")
        @test p.recall >= recall_lower_bound * 0.99  # not 1 to allow some "numerical" deviations
    end
end
