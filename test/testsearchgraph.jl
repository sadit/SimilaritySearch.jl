using SimilaritySearch, SimilaritySearch.AdjacencyLists, Random, StatsBase, Statistics
using Test, JET

#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function run_graph(G, ctx, queries, ksearch, gold_knns)
    @info typeof(G)
    searchtime = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    # @test_call searchbatch(G, ctx, queries, ksearch)
    recall = macrorecall(gold_knns, knns)
    @test recall >= 0.7
    @show recall, searchtime, length(queries) / searchtime
end


@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    n, m, dim = 100_000, 100, 8

    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))

    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, db)
    goldtime = @elapsed gold_knns = searchbatch(seq, getcontext(seq), queries, ksearch)

    #=
    @testset "fixed params" begin
        for bsize in [2, 12]
            algo = BeamSearch(; bsize)
            @info "=================== $algo"
            graph = SearchGraph(; db=DynamicMatrixDatabase(Float32, dim), dist, algo=algo)
            ctx = SearchGraphContext(
                neighborhood = Neighborhood(filter=IdentityNeighborhood()),
                hyperparameters_callback = OptimizeParameters(ParetoRecall()),
                parallel_block = 8
            )
            append_items!(graph, ctx, db)
            @test n == length(db) == length(graph)
            searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
            @test size(knns) == (ksearch, m) == size(gold_knns)
            @show goldD[:, 1]
            @show D[:, 1]
            @show gold_knns[:, 1]
            @show I[:, 1]
            recall = macrorecall(gold_knns, knns)
            @info "testing algo: $(string(graph.algo)), time: $(searchtime)"
            @test recall >= 0.6
            @info "queries per second: $(m/searchtime), recall: $recall"
            @info "===="
        end
    end
    =#

    @testset "AutoBS with ParetoRadius" begin
        graph = SearchGraph(; dist, algo=BeamSearch(bsize=2))
        ctx = SearchGraphContext(
            neighborhood = Neighborhood(filter=SatNeighborhood()),
            hyperparameters_callback = OptimizeParameters(OptRadius()),
            parallel_block = 8
        )
        #ctx = getcontext(graph)
        append_items!(graph, ctx, db)
        @test n == length(db) == length(graph)
        @info "---- starting ParetoRadius optimization ---"
        optimize_index!(graph, ctx, ParetoRadius())
        searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
        @test size(knns) == (ksearch, m) == size(gold_knns)
        recall = macrorecall(gold_knns, knns)
        @info "ParetoRadius:> queries per second: ", m/searchtime, ", recall:", recall
        @info graph.algo
        @test recall >= 0.6  # we don't expect high quality results on ParetoRadius

        @info "---- starting ParetoRecall optimization ---"
        optimize_index!(graph, ctx, ParetoRecall())
        searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
        @test size(knns) == (ksearch, m) == size(gold_knns)
        recall = macrorecall(gold_knns, knns)
        @info "ParetoRecall:> queries per second: ", m/searchtime, ", recall:", recall
        @info graph.algo
        @test recall >= 0.6
    end

    @info "========================= AutoBS MinRecall ======================"
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(
        neighborhood = Neighborhood(filter=SatNeighborhood()),
        hyperparameters_callback = OptimizeParameters(MinRecall(0.9)),
        parallel_block = 16
    )
    index!(graph, ctx)
    @test n == length(db) == length(graph)
    optimize_index!(graph, ctx, MinRecall(0.9); queries, ksearch)
    searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
    @test size(knns) == (ksearch, m) == size(gold_knns)
    recall = macrorecall(gold_knns, knns)
    @info "testing without additional optimizations: queries per second:", m/searchtime, ", recall: ", recall
    @info graph.algo
    @test recall >= 0.6
    
    @testset "rebuild" begin
        graph = rebuild(graph, ctx)
        @test n == length(db) == length(graph)
        optimize_index!(graph, ctx, MinRecall(0.9); queries)  # using the actual dataset makes prone to overfitting hyperparameters (more noticeable in rebuilt indexes)
        @info graph.algo, length(queries), ksearch
        searchtime_ = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
        @test size(knns) == (ksearch, m) == size(gold_knns)
        recall_ = macrorecall(gold_knns, knns)
        @test recall * 0.9 < recall_ # the rebuild should be pretty similar or better than the original one
        @info "-- old vs rebuild> searchtime: $searchtime vs $searchtime_; recall: $recall vs $recall_"
    end

    @testset "saveindex and loadindex" begin
        tmpfile = tempname()
        @info "--- load and save!!!"
        saveindex(tmpfile, graph; meta=[1, 2, 4, 8], store_db=false)
        let (G, meta) = loadindex(tmpfile, database(graph); staticgraph=true)
            @test G.adj isa StaticAdjacencyList
            @test length(G) == length(graph)
            @test length(G.adj) == length(graph.adj)
            @test distance(G) == distance(graph)
            @test database(G) === database(graph)
            @test G.hints == graph.hints

            for i in rand(eachindex(graph.adj), 100)
                @test neighbors(graph.adj, i) == neighbors(G.adj, i) 
                @test neighbors_length(graph.adj, i) == neighbors_length(G.adj, i) 
            end
            @test meta == [1, 2, 4, 8]
            @time run_graph(G, ctx, queries, ksearch, gold_knns)
        end
    end
    # exit(0)

    @info "#############=========== StrideMatrixDatabase with default parameters ==========###########"
    dim = 4
    m = 3000
    dimfake = dim * 1
    n = 10^5
    # dist = TurboSqL2Distance()
    dist = SqL2Distance()
    db = StrideMatrixDatabase(randn(Float32, dim, n))
    queries = StrideMatrixDatabase(randn(Float32, dim, m))
    #=
    db = let X = randn(Float32, dimfake, n)
        dim < dimfake && (X[dim+1:dimfake, :] .= 0f0)
        StrideMatrixDatabase(X)
    end
    queries = let X = randn(Float32, dimfake, n)
        dim < dimfake && (X[dim+1:dimfake, :] .= 0f0)
        StrideMatrixDatabase(X)
    end=#
    seq = ExhaustiveSearch(; dist, db)
    gold_knns = searchbatch(seq, getcontext(seq), queries, ksearch)
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(
        neighborhood = Neighborhood(filter=SatNeighborhood(), logbase=1.5),
        # neighborhood = Neighborhood(filter=IdentityNeighborhood(), logbase=1.5, connect_reverse_links_factor=0.8f0),
        hyperparameters_callback = OptimizeParameters(OptRadius(0.03), ksearch=ksearch+2),
        #hyperparameters_callback = OptimizeParameters(MinRecall(0.99)),
        parallel_block = 16
    )
    buildtime = @elapsed index!(graph, ctx)
    @test n == length(db) == length(graph)
    @test_call target_modules=(@__MODULE__,) search(graph, ctx, queries[1], knn(1))
    @test_call target_modules=(@__MODULE__,) searchbatch(graph, ctx, queries, ksearch)
    optimize_index!(graph, ctx, MinRecall(0.9))
    searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
    searchtime2 = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
    recall = macrorecall(gold_knns, knns)
    G = matrixhints(graph, StrideMatrixDatabase)
    searchtime3 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    searchtime4 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    recall_ = macrorecall(gold_knns, knns)
    searchtime5 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    mem = sum(map(length, graph.adj.end_point)) * sizeof(eltype(graph.adj.end_point[1])) / 2^20 # adj mem
    @info "buildtime: $buildtime sec, memory: $(mem)MB, recall: $recall, recall with AdjacentStoredHints: $recall_"
    @info "A> QpS: $(m/searchtime), QpS (already compiled): $(m/searchtime2)"
    @info "B> QpS: $(m/searchtime3), QpS (already compiled): $(m/searchtime5)"
    N = [neighbors_length(graph.adj, i) for i in eachindex(graph.adj)]
    @info quantile(N, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    @info graph.algo
    @test recall >= 0.7

    buildtime = @elapsed G = rebuild(graph, ctx)
    @time knns = searchbatch(G, ctx, queries, ksearch)
    recall_ = macrorecall(gold_knns, knns)
    searchtime5 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    mem = sum(map(length, G.adj.end_point)) * sizeof(eltype(G.adj.end_point[1])) / 2^20 # adj mem
    @info "rebuild buildtime: $buildtime sec, memory: $(mem)MB, recall: $recall => $recall_"
    @info "rebuild C> QpS (already compiled): $(m/searchtime5)"
    N = [neighbors_length(G.adj, i) for i in eachindex(G.adj)]
    @info quantile(N, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    @info graph.algo
    @test recall_ >= 0.7
end

