using SimilaritySearch
using SimilaritySearch.SimilarReferences
using Test


function test_vectors(create_index, dist::Function, ksearch, nick)
    @testset "indexing vectors with $nick and $dist" begin
        n = 1000 # number of items in the dataset
        m = 100  # number of queries
        dim = 3  # vector's dimension

        db = [rand(Float32, dim) |> normalize! for i in 1:n]
        queries = [rand(Float32, dim) |> normalize! for i in 1:m]

        index = create_index(db)
        optimize!(index, dist, recall=0.9, k=10)
        perf = Performance(dist, index.db, queries, expected_k=10)
        p = probe(perf, index, dist)
        @show dist, p
        @test p.recall > 0.8

        @info "adding more items"
        for item in queries
            push!(index, dist, item)
        end
        perf = Performance(dist, index.db, queries, expected_k=1)
        p = probe(perf, index, dist)
        @show dist, p
        @test p.recall > 0.999
        return p
    end
end

function test_sequences(create_index, dist::Function, ksearch, nick)
    @testset "indexing sequences with $nick and $dist" begin
        n = 1000 # number of items in the dataset
        m = 100  # number of queries
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
        index = create_index(db)
        # optimize!(index, recall=0.9, k=10)
        perf = Performance(dist, index.db, queries, expected_k=10)
        p = probe(perf, index, dist)
        @show dist, p
        @test p.recall > 0.1  ## Performance object tests object identifiers, but sequence distances have a lot of distance collisions

        # for item in queries
        #     push!(index, dist, item)
        # end
        # perf = Performance(dist, index.db, queries, expected_k=1)
        # p = probe(perf, index, dist)
        # @show dist, p
        # @test p.recall > 0.999
        # return p
    end
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    σ = 127
    κ = 3

    for dist in [
        l2_distance, # 1.0 -> metric, < 1.0 if dist is not a metric
        l1_distance,
        linf_distance,
        lp_distance(3),
        lp_distance(0.5),
        angle_distance
    ]
        p = test_vectors((db) -> fit(Knr, dist, db, numrefs=σ, k=κ), dist, ksearch, "KNR")
    end
end

@testset "indexing sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    σ = 127
    κ = 3
    
    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        jaccard_distance,
        dice_distance,
        intersection_distance,
        common_prefix_distance,
        levenshtein_distance,
        lcs_distance,
        hamming_distance,
    ]   
        p = test_sequences((db) -> fit(Knr, dist, db, numrefs=σ, k=κ), dist, ksearch, "KNR")
    end
end
