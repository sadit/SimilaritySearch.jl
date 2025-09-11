using SimilaritySearch, SimilaritySearch.AdjacencyLists, Random, StatsBase, Statistics
using Test, JET
using AllocCheck
#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function check_graph(G, ctx, queries, ksearch)
    res = knnqueue(ctx, ksearch)
    @test_opt search(G, ctx, queries[2], res)
    #@test_call target_modules=(@__MODULE__,) search(G, ctx, queries[2], res)

    # knns = xknnpool(ksearch, length(queries))
    @check_allocs function do_something()
        search(G.algo, G, ctx, queries[2], res, G.hints)
        #searchbatch!(G, ctx, queries, knns)
    end

    try
        do_something()
    catch err
        for (i, e) in enumerate(err.errors)
            display("=============== $i ===========")
            display(e)
        end
    end

    exit(0)
end

function prepare_benchmark(Database;
        ksearch::Int = 8,
        n::Int=10^5,
        m::Int=10^3,
        dim::Int=8)
    
    db = Database(rand(Float32, dim, n))
    queries = Database(rand(Float32, dim, m))

    dist = SqL2Distance()
    seq = ExhaustiveSearch(; dist, db)
    ectx = GenericContext()

    @time searchbatch(seq, ectx, queries, ksearch)
    searchtime = @elapsed gold_knns = searchbatch(seq, ectx, queries, ksearch)

    B = (; dist, db, queries, ksearch, n, m, dim, gold=(; knns=gold_knns, searchtime))

    let res = knnqueue(ectx, ksearch), q = queries[2], ectx=ectx, seq=seq
        @test_call target_modules=(@__MODULE__,) search(seq, ectx, queries[2], res)
        @time "SEARCH Exhaustive 1" search(seq, ectx, q, res)
        @time "SEARCH Exhaustive 2" search(seq, ectx, q, res)
        # @code_warntype search(seq, ectx, q, res)

        f(seq, ectx, q, res) = @time "SEARCH Exhaustive 3" search(seq, ectx, q, res)
        f(seq, ectx, q, res)
        @show typeof(seq) typeof(ectx) typeof(q) typeof(res)
        search(seq, ectx, q, res)
        #=
        @check_allocs function do_something(seq, ectx, q, res)
            reuse!(res)
            knns = search(seq, ectx, q, res)
        end

        try
            do_something(seq, ectx, q, res)
            do_something(seq, ectx, q, res)
        catch err
            for (i, e) in enumerate(err.errors)
                display("=============== $i ===========")
                display(e)
            end
        end=#

    end

    B
end

function abs_minrecall(B)
    @info "===================== minrecall =============================="
    graph = SearchGraph(; B.db, B.dist)
    ctx = SearchGraphContext(
        neighborhood = Neighborhood(filter=SatNeighborhood()),
        #neighborhood = Neighborhood(filter=IdentityNeighborhood()),
        hyperparameters_callback = OptimizeParameters(MinRecall(0.99)),
        verbose=false
    )

    index!(graph, ctx)

    @show quantile(neighbors_length.(Ref(graph.adj), 1:length(graph)), 0:0.1:1.0)
    @test B.n == length(B.db) == length(graph)
    optimize_index!(graph, ctx, MinRecall(0.9); B.queries, B.ksearch)
    searchtime = @elapsed knns = searchbatch(graph, ctx, B.queries, B.ksearch)
    @test size(knns) == (B.ksearch, B.m) == size(B.gold.knns)
    recall = macrorecall(B.gold.knns, knns)
    @info "minrecall: queries per second: $(B.m/searchtime), recall: $(recall)"
    @show graph.algo
    @show quantile(neighbors_length.(Ref(graph.adj), 1:length(graph)), 0:0.1:1.0)
    @test recall >= 0.8

    
    graph, ctx
end

function abs_rebuild(graph, ctx, B)
    @info "===================== rebuild =============================="
    graph = rebuild(graph, ctx)
    @test B.n == length(B.db) == length(graph)
    optimize_index!(graph, ctx, MinRecall(0.9); B.queries)  # using the actual dataset makes prone to overfitting hyperparameters (more noticeable in rebuilt indexes)
    @show graph.algo, length(B.queries), B.ksearch
    searchtime = @elapsed knns = searchbatch(graph, ctx, B.queries, B.ksearch)
    @test size(knns) == (B.ksearch, B.m) == size(B.gold.knns)
    recall = macrorecall(B.gold.knns, knns)
    @info "rebuild: queries per second: $(B.m/searchtime), recall: $(recall)"
    @show graph.algo
    @show quantile(neighbors_length.(Ref(graph.adj), 1:length(graph)), 0:0.1:1.0)
    @test recall >= 0.8
end

function abs_save_and_load(graph, ctx, B)
    @info "===================== saveindex and loadindex StaticAdjacentList Graph ==============="
    tmpfile = tempname()
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
        @test_call target_modules=(@__MODULE__,) searchbatch(G, ctx, B.queries, B.ksearch)

        knns = zeros(IdWeight, B.ksearch, length(B.queries))
        @time knns = searchbatch!(G, ctx, B.queries, knns)
        searchtime = @elapsed knns = searchbatch!(G, ctx, B.queries, knns)
        recall = macrorecall(B.gold.knns, knns)

        @info "loaded: queries per second: $(B.m/searchtime), recall: $(recall)"
        @show G.algo
        @show quantile(neighbors_length.(Ref(G.adj), 1:length(G)), 0:0.1:1.0)
        @test recall >= 0.8
    end
end

function abs_matrixhints(graph, ctx, B, _Database)
    @info "===================== matrixhints =============================="
    graph = matrixhints(graph, _Database)
    @test B.n == length(B.db) == length(graph)
    optimize_index!(graph, ctx, MinRecall(0.9); B.queries)  # using the actual dataset makes prone to overfitting hyperparameters (more noticeable in rebuilt indexes)
    @show graph.algo, length(B.queries), B.ksearch
    knns = zeros(IdWeight, B.ksearch, length(B.queries))
    @time knns = searchbatch!(graph, ctx, B.queries, knns)
    searchtime = @elapsed searchbatch!(graph, ctx, B.queries, knns)
    @test size(knns) == (B.ksearch, B.m) == size(B.gold.knns)
    recall = macrorecall(B.gold.knns, knns)
    @info "matrixhints: queries per second: $(B.m/searchtime), recall: $(recall)"
    @show graph.algo
    @show quantile(neighbors_length.(Ref(graph.adj), 1:length(graph)), 0:0.1:1.0)
    @test recall >= 0.8
end

@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required

    @testset "MatrixDatabase" begin
        B = prepare_benchmark(MatrixDatabase)
        graph, ctx = abs_minrecall(B)
        abs_rebuild(graph, ctx, B)
        abs_save_and_load(graph, ctx, B)
        abs_matrixhints(graph, ctx, B, MatrixDatabase)
    end

    @testset "StrideMatrixDatabase" begin
        B = prepare_benchmark(StrideMatrixDatabase)
        graph, ctx = abs_minrecall(B)
        abs_rebuild(graph, ctx, B)
        abs_save_and_load(graph, ctx, B)
        abs_matrixhints(graph, ctx, B, StrideMatrixDatabase)
    end


    #@test_call target_modules=(@__MODULE__,) search(graph, ctx, queries[1], knn(1))
    #@test_call target_modules=(@__MODULE__,) searchbatch(graph, ctx, queries, ksearch)
    
    
    #=@testset "AutoBS with ParetoRadius" begin
        graph = SearchGraph(; dist, algo=BeamSearch(bsize=2))
        ctx = SearchGraphContext(
            neighborhood = Neighborhood(filter=SatNeighborhood()),
            hyperparameters_callback = OptimizeParameters(OptRadius()),
            parallel_block = 8
        )
        #ctx = getcontext(graph)
        try
            append_items!(graph, ctx, db)
        catch err
            display(err.errors[1])
            exit(0)
        end
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
    =#

end

