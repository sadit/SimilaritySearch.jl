using SimilaritySearch
using Test

function test_binhamming(create_index, dist::Function, ksearch, nick, create)
    @testset "indexing vectors with $nick with hamming_distance" begin
        n = 300 # number of items in the dataset
        m = 30  # number of queries

        db = [create() for i in 1:n]
        queries = [create() for i in 1:m]

        index = create_index(db)
        @test length(index.db) == n
        push!(index, dist, create())
        @test length(index.db) == 1 + n
        perf = Performance(dist, index.db, queries, expected_k=10)
        p = probe(perf, index, dist)
        @show p
        return p
    end
end

function test_cos(create_index, dist::Function, ksearch, nick; repeat=1, aggregation=:mean)
    @testset "indexing vectors with $nick with cos or angle's distance" begin
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
        perf = Performance(dist, index.db, [@view queries[:, i] for i in 1:size(queries, 2)], expected_k=10)
        p = probe(perf, index, dist, repeat=repeat, aggregation=aggregation)
        @show p
        return p
    end
end

function test_vectors(create_index, dist, ksearch, nick; repeat=1, aggregation=:mean)
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

function test_sequences(create_index, dist, ksearch, nick)
    @testset "indexing sequences with $nick and $dist" begin
        n = 300 # number of items in the dataset
        m = 30  # number of queries
        dim = 5  # the length of sequences
        V = collect(1:10)  # vocabulary of the sequences

        function create_item()
            s = rand(V, dim)
            if dist == jaccard_distance || dist == dice_distance || dist == intersection_distance
                sort!(s)
                s = unique(s)
            end

            return s
        end
        db = [create_item() for i in 1:n]
        queries = [create_item() for i in 1:m]

        @info "inserting items into the index"
        # index = Laesa(db, dist, σ)
        index = create_index(db)
        @test length(index.db) == n
        perf = Performance(dist, index.db, queries, expected_k=10)
        p = probe(perf, index, dist)
        # @show dist, p
        return p
    end
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10

    for (recall_lower_bound, dist) in [
        (1.0, l2_distance), # 1.0 -> metric, < 1.0 if dist is not a metric
        (1.0, l1_distance),
        (1.0, linf_distance),
        (0.1, squared_l2_distance),
        (1.0, lp_distance(3.0)),
        (0.1, lp_distance(0.5))
    ]
        @show recall_lower_bound, dist
        p = test_vectors((db) -> fit(Sequential, db), dist, ksearch, "Sequential")
        @test p.recall >= recall_lower_bound * 0.99 # to support "numerical" variations

        p = test_vectors((db) -> fit(Laesa, dist, db, 8), dist, ksearch, "Laesa")
        @test p.recall >= recall_lower_bound * 0.99
    end
end

@testset "indexing sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10

    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for (recall_lower_bound, dist) in [
        (1.0, jaccard_distance),
        (0.1, dice_distance),
        (0.1, intersection_distance),
        (0.1, common_prefix_distance),
        (1.0, levenshtein_distance),
        (1.0, lcs_distance),
        (1.0, hamming_distance)
    ]
        p = test_sequences((db) -> fit(Sequential, db), dist, ksearch, "Sequential")
        @test p.recall >= recall_lower_bound * 0.99  # not 1 to allow some "numerical" deviations
        p = test_sequences((db) -> fit(Laesa, dist, db, 1), dist, ksearch, "Laesa")
        @test p.recall >= recall_lower_bound * 0.99  # not 1 to allow some "numerical" deviations
    end
end

@testset "misc" begin
    # cosine and angle distance
    ksearch = 10

    p1 = test_cos((db) -> fit(Sequential, db), angle_distance, ksearch, "Sequential", repeat=3, aggregation=:median)
    p2 = test_cos((db) -> fit(Sequential, db), angle_distance, ksearch, "Sequential", repeat=3, aggregation=:min)
    p3 = test_cos((db) -> fit(Sequential, db), cosine_distance, ksearch, "Sequential", repeat=3, aggregation=:max)
    p4 = test_cos((db) -> fit(Sequential, db), cosine_distance, ksearch, "Sequential", repeat=3, aggregation=:mean)
    @show p1, p2, p3, p4
    @test p1.recall > 0.99
    @test p2.recall > 0.99
    @test p3.recall > 0.99
    @test p4.recall > 0.99

    p = test_binhamming((db) -> Sss(hamming_distance, db, 0.35), hamming_distance, ksearch, "Sequential", () -> UInt32[rand(UInt32) for i in 1:7])
    @test p.recall > 0.99

    p = test_binhamming((db) -> LaesaTournament(hamming_distance, db, 16, 3), hamming_distance, ksearch, "Sequential", () -> rand(UInt64))
    @test p.recall > 0.99
end
