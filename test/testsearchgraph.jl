
using SimilaritySearch, SimilaritySearch.AdjacencyLists, Random
using Test, JET

#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function run_graph(G, queries, ksearch, Igold)
    ctx = getcontext(G)
    searchtime = @elapsed I, _ = searchbatch(G, ctx, queries, ksearch)
    @test_call searchbatch(G, ctx, queries, ksearch)
    recall = macrorecall(Igold, I)
    @test recall >= 0.7
    @show recall, searchtime, 1 / searchtime
end


@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    n, m, dim = 100_000, 100, 8

    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))

    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, db)
    goldtime = @elapsed goldI, goldD = searchbatch(seq, getcontext(seq), queries, ksearch)

    #=
    @testset "fixed params" begin
        for bsize in [2, 12]
            search_algo = BeamSearch(; bsize)
            @info "=================== $search_algo"
            graph = SearchGraph(; db=DynamicMatrixDatabase(Float32, dim), dist, search_algo=search_algo)
            ctx = SearchGraphContext(
                neighborhood = Neighborhood(reduce=IdentityNeighborhood()),
                hyperparameters_callback = OptimizeParameters(ParetoRecall()),
                parallel_block = 8
            )
            append_items!(graph, ctx, db)
            @test n == length(db) == length(graph)
            searchtime = @elapsed I, D = searchbatch(graph, ctx, queries, ksearch)
            @test size(I) == size(D) == (ksearch, m) == size(goldI)
            @show goldD[:, 1]
            @show D[:, 1]
            @show goldI[:, 1]
            @show I[:, 1]
            recall = macrorecall(goldI, I)
            @info "testing search_algo: $(string(graph.search_algo)), time: $(searchtime)"
            @test recall >= 0.6
            @info "queries per second: $(m/searchtime), recall: $recall"
            @info "===="
        end
    end
    =#

    @testset "AutoBS with ParetoRadius" begin
        graph = SearchGraph(; dist, search_algo=BeamSearch(bsize=2))
        ctx = SearchGraphContext(getcontext(graph);
            neighborhood = Neighborhood(reduce=SatNeighborhood()),
            hyperparameters_callback = OptimizeParameters(OptRadius()),
            parallel_block = 8
        )
        #ctx = getcontext(graph)
        append_items!(graph, ctx, db)
        @test n == length(db) == length(graph)
        @info "---- starting ParetoRadius optimization ---"
        optimize_index!(graph, ctx, ParetoRadius())
        searchtime = @elapsed I, D = searchbatch(graph, ctx, queries, ksearch)
        @test size(I) == size(D) == (ksearch, m) == size(goldI)
        recall = macrorecall(goldI, I)
        @info "ParetoRadius:> queries per second: ", m/searchtime, ", recall:", recall
        @info graph.search_algo
        @test recall >= 0.6  # we don't expect high quality results on ParetoRadius

        @info "---- starting ParetoRecall optimization ---"
         optimize_index!(graph, ctx, ParetoRecall())
        searchtime = @elapsed I, D = searchbatch(graph, ctx, queries, ksearch)
        @test size(I) == size(D) == (ksearch, m) == size(goldI)
        recall = macrorecall(goldI, I)
        @info "ParetoRecall:> queries per second: ", m/searchtime, ", recall:", recall
        @info graph.search_algo
        @test recall >= 0.6
    end

    @info "========================= AutoBS MinRecall ======================"
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(getcontext(graph);
        neighborhood = Neighborhood(reduce=SatNeighborhood()),
        hyperparameters_callback = OptimizeParameters(MinRecall(0.9)),
        parallel_block = 16
    )
    index!(graph, ctx)
    @test n == length(db) == length(graph)
    optimize_index!(graph, ctx, MinRecall(0.9); queries)
    searchtime = @elapsed I, D = searchbatch(graph, ctx, queries, ksearch)
    @test size(I) == size(D) == (ksearch, m) == size(goldI)
    recall = macrorecall(goldI, I)
    @info "testing without additional optimizations: queries per second:", m/searchtime, ", recall: ", recall
    @info graph.search_algo
    @test recall >= 0.6
    
    @testset "rebuild" begin
        graph = rebuild(graph, ctx)
        @test n == length(db) == length(graph)
        optimize_index!(graph, ctx, MinRecall(0.9); queries)  # using the actual dataset makes prone to overfitting hyperparameters (more noticeable in rebuilt indexes)
        searchtime_ = @elapsed I, D = searchbatch(graph, ctx, queries, ksearch)
        @test size(I) == size(D) == (ksearch, m) == size(goldI)
        recall_ = macrorecall(goldI, I)

        @info "-- old vs rebuild> searchtime: $searchtime vs $searchtime_; recall: $recall vs $recall_"
    end

    @testset "saveindex and loadindex" begin
        tmpfile = tempname()
        @info "--- load and save!!!"
        saveindex(tmpfile, graph; meta=[1, 2, 4, 8], store_db=false)
        let
            G, meta = loadindex(tmpfile, database(graph); staticgraph=true)
            @test meta == [1, 2, 4, 8]
            @test G.adj isa StaticAdjacencyList
            @time run_graph(G, queries, ksearch, goldI)
        end
    end

    @info "#############=========== StrideMatrixDatabase with default parameters ==========###########"
    dim = 4
    db = StrideMatrixDatabase(randn(Float32, dim, n))
    queries = StrideMatrixDatabase(randn(Float32, dim, m))
    seq = ExhaustiveSearch(; dist, db)
    goldI, goldD = searchbatch(seq, ctx, queries, ksearch)
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(getcontext(graph);
        neighborhood = Neighborhood(reduce=SatNeighborhood(), logbase=2.0),
        hyperparameters_callback = OptimizeParameters(OptRadius(0.001)),
        parallel_block = 16
    )
    buildtime = @elapsed index!(graph, ctx)
    @test n == length(db) == length(graph)
    @test_call search(graph, ctx, queries[1], KnnResult(1))
    @test_call searchbatch(graph, ctx, queries, ksearch)
    optimize_index!(graph, ctx, MinRecall(0.8))
    searchtime = @elapsed I, _ = searchbatch(graph, ctx, queries, ksearch)
    searchtime2 = @elapsed I, _ = searchbatch(graph, ctx, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "buildtime", buildtime
    @info "testing without additional optimizations> queries per second (including compilation): ", m/searchtime, ", searchtime2 (already compiled):", m/searchtime2, ", recall: ", recall
    @info graph.search_algo
    @test recall >= 0.7
end

