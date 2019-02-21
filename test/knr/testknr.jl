using SimilaritySearch
using SimilaritySearch.SimilarReferences
using Test


function test_vectors(create_index, dist, ksearch, nick)
    @testset "indexing vectors with $nick and $dist" begin
        n = 1000 # number of items in the dataset
        m = 100  # number of queries
        dim = 3  # vector's dimension

        db = [rand(Float32, dim) for i in 1:n]
        queries = [rand(Float32, dim) for i in 1:m]

        index = create_index(db)
        optimize!(index, recall=0.9, k=10, use_distances=false)
        perf = Performance(index.db, dist, queries, expected_k=10)
        p = probe(perf, index, use_distances=false)
        @show dist, p
        @test p.recall > 0.8

        @info "adding more items"
        for item in queries
            push!(index, item)
        end
        perf = Performance(index.db, dist, queries, expected_k=1)
        p = probe(perf, index, use_distances=false)
        @show dist, p
        @test p.recall > 0.999
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

        @info "inserting items into the index"
        index = create_index(db)
        # optimize!(index, recall=0.9, k=10, use_distances=true)
        perf = Performance(index.db, dist, queries, expected_k=10)
        p = probe(perf, index, use_distances=true)
        @show dist, p
        @test p.recall > 0.6

        for item in queries
            push!(index, item)
        end
        perf = Performance(index.db, dist, queries, expected_k=1)
        p = probe(perf, index, use_distances=true)
        @show dist, p
        @test p.recall > 0.999
        return p
    end
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    σ = 127
    κ = 3

    for dist in [
        L2Distance(), # 1.0 -> metric, < 1.0 if dist is not a metric
        L1Distance(),
        LInfDistance(),
        # AngleDistance(),
        LpDistance(3),
        LpDistance(0.5)
    ]
        p = test_vectors((db) -> Knr(db, dist, numrefs=σ, k=κ), dist, ksearch, "KNR")
    end
end

@testset "indexing sequences" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    σ = 127
    κ = 3
    
    # metric distances should achieve recall=1 (perhaps lesser because of numerical inestability)
    for dist in [
        JaccardDistance(),
        DiceDistance(),
        IntersectionDistance(),
        CommonPrefixDistance(),
        LevDistance(),
        LcsDistance(),
        HammingDistance(),
    ]   
        p = test_sequences((db) -> Knr(db, dist, numrefs=σ, k=κ), dist, ksearch, "KNR")
    end
end
