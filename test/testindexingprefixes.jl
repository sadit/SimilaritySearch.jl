# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Random, Test

@testset "index! by :knr" begin
    dim, n, m = 8, 5_000, 30
    numrefs = 256
    hints_size = ceil(Int, sqrt(n))
    ksearch = 8
    ksize = 8
    n_neighbors = 4

    dist = Dist.SqL2()

    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))
    ctx = SearchGraphContext()

    seq = ExhaustiveSearch(dist, db)
    gold_knns_ids, gold_knns_dists = searchbatch(seq, GenericContext(), queries, ksearch)

    @testset "index! :knr with :fft" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :knr; numrefs, k=ksize, sample_method=:fft, hints_size, n_neighbors, start_factor=0.97)
        @test length(graph) == n
        @test distance(graph) == dist

        #@time "rebuild" graph = rebuild(graph, ctx)
        #optimize_index!(graph, ctx, MinRecall(0.85); queries, ksearch)
        @time "search" knns_ids, knns_dists = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns_ids, knns_ids)
        @info "knr (:fft + rebuild) recall: $recall"
        @test recall >= 0.7

        @testset "directcount on a :knr-built graph (no reverse/direct distinction)" begin
            # :knr never calls connect_reverse_links! -- every edge is "direct" by construction
            @test all(graph.directcount[i] == neighbors_length(graph.adj, i) for i in eachindex(graph.adj))
            # destructive -- must be last, nothing after this relies on `graph` staying intact
            @test_logs (:warn, r"empty the entire graph") remove_direct_links!(graph)
            @test all(iszero, graph.directcount)
            @test all(iszero(neighbors_length(graph.adj, i)) for i in eachindex(graph.adj))
        end
    end

    @testset "index! :knr with :random" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :knr; numrefs, k=ksize, sample_method=:random, hints_size, n_neighbors, start_factor=0.97)
        @test length(graph) == n
        @test distance(graph) == dist

        #@time "rebuild" graph = rebuild(graph, ctx)
        #optimize_index!(graph, ctx, MinRecall(0.85); queries, ksearch)
        @time "search" knns_ids, knns_dists = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns_ids, knns_ids)
        @info "knr (:random + rebuild) recall: $recall"
        @test recall >= 0.7
        @test all(graph.directcount[i] == neighbors_length(graph.adj, i) for i in eachindex(graph.adj))
    end
end
