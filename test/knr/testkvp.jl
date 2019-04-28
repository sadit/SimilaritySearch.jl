using SimilaritySearch
using SimilaritySearch.SimilarReferences
using Test


function test_vectors(create_index, dist::Function, ksearch, nick)
    @testset "indexing vectors with $nick and $dist" begin
        n = 300 # number of items in the dataset
        m = 30  # number of queries
        dim = 3  # vector's dimension

        db = [rand(Float32, dim) for i in 1:n]
        queries = [rand(Float32, dim) for i in 1:m]

        index = create_index(db)
        # testing pushes
        push!(index, dist, rand(Float32, dim))
        @test length(index.db) == n + 1
        perf = Performance(dist, index.db, queries, expected_k=10)
        p = probe(perf, index, dist)
        @show dist, p
        return p
    end
end

function test_sequences(create_index, dist::Function, ksearch, nick)
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
        # optimize!(index, recall=0.9, use_distances=true)
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
    local index

    for (recall_lower_bound, dist) in [
        (1.0, l2_distance), # 1.0 -> metric, < 1.0 if dist is not a metric
        (1.0, l1_distance),
        (1.0, linf_distance),
        (1.0, lp_distance(3.0)),
        (0.1, lp_distance(0.5))
    ]
        @show recall_lower_bound, dist

        @show Kvp
        p = test_vectors((db) -> fit(Kvp, dist, db, 3, 32), dist, ksearch, "Kvp")
        @test p.recall >= recall_lower_bound * 0.99 # not 1 to allow some "numerical" deviations
    end
end

@testset "indexing sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    local index

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
        p = test_sequences((db) -> fit(Kvp, dist, db, 3, 32), dist, ksearch, "Kvp")
        @test p.recall >= recall_lower_bound * 0.99  # not 1 to allow some "numerical" deviations
    end
end
