# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Random, Test

@testset "index! by :prefixes" begin
    dim, n, m = 8, 10000, 100
    ksearch = 12
    dist = Dist.SqL2()

    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))
    ctx = SearchGraphContext()

    seq = ExhaustiveSearch(dist, db)
    gold_knns = searchbatch(seq, GenericContext(), queries, ksearch)


    @testset "SearchGraph base" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx)
        @test length(graph) == n
        @test distance(graph) == dist

        optimize_index!(graph, ctx, MinRecall(0.9); queries, ksearch)
        knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "plain SearchGraph recall: $recall"
        @test recall >= 0.8
    end

    @testset "index! :prefixes with :fft" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :prefixes; numrefs=32, k=4, sample_method=:fft, probfactor=0.9)
        @test length(graph) == n
        @test distance(graph) == dist

        optimize_index!(graph, ctx, MinRecall(0.9); queries, ksearch)
        knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "prefixes (:fft) recall: $recall"
        @test recall >= 0.8
    end

    @testset "index! :prefixes with :random" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :prefixes; numrefs=32, k=4, sample_method=:random, probfactor=0.9)
        @test length(graph) == n
        @test distance(graph) == dist

        optimize_index!(graph, ctx, MinRecall(0.9); queries, ksearch)
        knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "prefixes (:random) recall: $recall"
        @test recall >= 0.8
    end
end
