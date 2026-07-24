# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Random, Test

@testset "index! by :prefixes" begin
    dim, n, m = 4, 10000, 100
    numrefs = ceil(Int, sqrt(n))
    ksearch = 12
    hints_size = numrefs ÷ 2
    ksize = 4
    max_cluster_full_link = 100
    comb_list = [2] # linked to ksize of the knr and should be used to achieve the desired threadoff

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
        @time "search" knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "plain SearchGraph recall: $recall"
        @test recall >= 0.8
    end

    @testset "index! :prefixes with :fft" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :knr; numrefs, k=ksize, sample_method=:fft, hints_size, max_cluster_full_link, comb_list)
        @test length(graph) == n
        @test distance(graph) == dist

        optimize_index!(graph, ctx, MinRecall(0.9); queries, ksearch)
        @time "search" knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "prefixes (:fft) recall: $recall"
        @test recall >= 0.8
    end

    @testset "index! :prefixes with :random" begin
        graph = SearchGraph(dist, db)
        @time "Graph construction" index!(graph, ctx, :knr; numrefs, k=ksize, sample_method=:random, hints_size, max_cluster_full_link, comb_list)
        @test length(graph) == n
        @test distance(graph) == dist

        optimize_index!(graph, ctx, MinRecall(0.9); queries, ksearch)
        @time "search" knns = searchbatch(graph, ctx, queries, ksearch)
        recall = macrorecall(gold_knns, knns)
        @info "prefixes (:random) recall: $recall"
        @test recall >= 0.8
    end
end
