# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Random, Test

@testset "index! by :knr" begin
    dim, n, m = 8, 10_000, 100
    numrefs = 512
    hints_size = ceil(Int, sqrt(n))
    ksearch = 8
    ksize = 8
    n_neighbors = 4

    dist = Dist.SqL2()

    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))
    ctx = SearchGraphContext()

    seq = ExhaustiveSearch(dist, db)
    gold_knns = searchbatch(seq, GenericContext(), queries, ksearch)

    @testset "index! :knr with :fft" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :knr; numrefs, k=ksize, sample_method=:fft, hints_size, n_neighbors, start_factor=0.97)
        @test length(graph) == n
        @test distance(graph) == dist

        #@time "rebuild" graph = rebuild(graph, ctx)
        #optimize_index!(graph, ctx, MinRecall(0.85); queries, ksearch)
        @time "search" knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "knr (:fft + rebuild) recall: $recall"
        @test recall >= 0.7
    end

    @testset "index! :knr with :random" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :knr; numrefs, k=ksize, sample_method=:random, hints_size, n_neighbors, start_factor=0.97)
        @test length(graph) == n
        @test distance(graph) == dist

        #@time "rebuild" graph = rebuild(graph, ctx)
        #optimize_index!(graph, ctx, MinRecall(0.85); queries, ksearch)
        @time "search" knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "knr (:random + rebuild) recall: $recall"
        @test recall >= 0.7
    end
end
