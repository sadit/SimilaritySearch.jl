using SimilaritySearch, Random, StatsBase, Statistics
using Test
#using AllocCheck

#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function prepare_benchmark(Database;
    ksearch::Int=8,
    n::Int=2_000,
    m::Int=30,
    dim::Int=4)

    db = Database(rand(Float32, dim, n))
    queries = Database(rand(Float32, dim, m))

    dist = Dist.SqL2()
    seq = ExhaustiveSearch(dist, db)
    ectx = GenericContext()

    @time searchbatch(seq, ectx, queries, ksearch)
    searchtime = @elapsed gold_knns_ids, gold_knns_dists = searchbatch(seq, ectx, queries, ksearch)

    B = (; dist, db, queries, ksearch, n, m, dim, gold=(; ids=gold_knns_ids, dists=gold_knns_dists, searchtime))

    let res = knnqueue(ectx, ksearch), q = queries[2], ectx = ectx, seq = seq
        #@test_call target_modules = (@__MODULE__,) search(seq, ectx, queries[2], res)
        @time "SEARCH Exhaustive 1" search(seq, ectx, q, res)
        @time "SEARCH Exhaustive 2" search(seq, ectx, q, res)
        # @code_warntype search(seq, ectx, q, res)

        f(seq, ectx, q, res) = @time "SEARCH Exhaustive 3" search(seq, ectx, q, res)
        f(seq, ectx, q, res)
        @show typeof(seq) typeof(ectx) typeof(q) typeof(res)
        search(seq, ectx, q, res)

    end

    B
end

function abs_minrecall(B; kwargs...)
    @info "===================== minrecall $kwargs =============================="
    graph = SearchGraph(B.dist, B.db; kwargs...)
    ctx = SearchGraphContext(
        neighborhood=Neighborhood(filter=SatNeighborhood()),
        #neighborhood = Neighborhood(filter=IdentityNeighborhood()),
        hyperparameters_callback=OptimizeParameters(MinRecall(0.99)),
        verbose=false
    )

    index!(graph, ctx)
    @show length(graph.adj), length(graph), length(B.db)
    @assert length(graph) == length(B.db) "length(graph) == length(B.db)"

    @show quantile(neighbors_length.(Ref(graph.adj), 1:length(graph)), 0:0.1:1.0)
    @test B.n == length(B.db) == length(graph)
    optimize_index!(graph, ctx, MinRecall(0.9); B.queries, B.ksearch)
    searchtime = @elapsed knns_ids, _ = searchbatch(graph, ctx, B.queries, B.ksearch)
    @test size(knns_ids) == (B.ksearch, B.m) == size(B.gold.ids)
    recall = macrorecall(B.gold.ids, knns_ids)
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
    searchtime = @elapsed knns_ids, _ = searchbatch(graph, ctx, B.queries, B.ksearch)
    @test size(knns_ids) == (B.ksearch, B.m) == size(B.gold.ids)
    recall = macrorecall(B.gold.ids, knns_ids)
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
        @test G.adj isa StaticAdjList
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
        #@test_call target_modules = (@__MODULE__,) searchbatch(G, ctx, B.queries, B.ksearch)

        @time knns_ids, _ = searchbatch(G, ctx, B.queries, B.ksearch)
        searchtime = @elapsed knns_ids, _ = searchbatch(G, ctx, B.queries, B.ksearch)
        recall = macrorecall(B.gold.ids, knns_ids)

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
    @time knns_ids, _ = searchbatch(graph, ctx, B.queries, B.ksearch)
    searchtime = @elapsed knns_ids, _ = searchbatch(graph, ctx, B.queries, B.ksearch)
    @test size(knns_ids) == (B.ksearch, B.m) == size(B.gold.ids)
    recall = macrorecall(B.gold.ids, knns_ids)
    @info "matrixhints: queries per second: $(B.m/searchtime), recall: $(recall)"
    @show graph.algo
    @show quantile(neighbors_length.(Ref(graph.adj), 1:length(graph)), 0:0.1:1.0)
    @test recall >= 0.8
end

@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required

    B = prepare_benchmark(MatrixDatabase)
    @testset "MatrixDatabase" begin

        graph, ctx = abs_minrecall(B)
        abs_rebuild(graph, ctx, B)
        #abs_save_and_load(graph, ctx, B)
        abs_matrixhints(graph, ctx, B, MatrixDatabase)
    end

    @testset "AdjDict" begin
        graph, ctx = abs_minrecall(B; adj=AdjDict(UInt32))
        abs_rebuild(graph, ctx, B)
        #abs_save_and_load(graph, ctx, B)
        abs_matrixhints(graph, ctx, B, MatrixDatabase)
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

@testset "direct/reverse neighbor bookkeeping (directcount)" begin
    dist = Dist.SqL2()
    n = 300
    db = MatrixDatabase(rand(Float32, 4, n))
    ctx = SearchGraphContext()

    @testset "directcount matches direct/reverse split" begin
        graph = SearchGraph(dist, db)
        index!(graph, ctx)

        for i in eachindex(graph.adj)
            full = neighbors(graph.adj, i)
            d = direct_neighbors(graph, i)
            r = reverse_neighbors(graph, i)
            @test length(d) == graph.directcount[i]
            @test length(d) + length(r) == neighbors_length(graph.adj, i) == length(full)
            @test collect(d) == full[1:graph.directcount[i]]
            @test collect(r) == full[graph.directcount[i]+1:end]
        end

        # on a graph this size, at least one node should have received a reverse edge
        @test any(graph.directcount[i] < neighbors_length(graph.adj, i) for i in eachindex(graph.adj))
    end

    @testset "remove_reverse_links! keeps only direct neighbors" begin
        graph = SearchGraph(dist, db)
        index!(graph, ctx)
        origdirect = copy(graph.directcount)

        remove_reverse_links!(graph)
        for i in eachindex(graph.adj)
            @test neighbors_length(graph.adj, i) == origdirect[i]
            @test graph.directcount[i] == origdirect[i]
        end

        # sanity: search still runs on the pruned graph (not a recall assertion)
        res = knnqueue(ctx, 5)
        search(graph, ctx, db[1], res)
        @test length(res) > 0
    end

    @testset "remove_direct_links! keeps only reverse neighbors" begin
        graph = SearchGraph(dist, db)
        index!(graph, ctx)
        reversecounts = [neighbors_length(graph.adj, i) - graph.directcount[i] for i in eachindex(graph.adj)]

        remove_direct_links!(graph)
        for i in eachindex(graph.adj)
            @test neighbors_length(graph.adj, i) == reversecounts[i]
            @test graph.directcount[i] == 0
        end
    end

    @testset "remove_direct_links! warns when no node has any reverse edges" begin
        graph = SearchGraph(dist, MatrixDatabase(rand(Float32, 4, 1)))
        index!(graph, ctx)  # a single node ends up with zero neighbors of any kind
        @test_logs (:warn, r"empty the entire graph") remove_direct_links!(graph)
    end

    @testset "rebuild produces a correctly-populated directcount" begin
        graph = SearchGraph(dist, db)
        index!(graph, ctx)
        graph = rebuild(graph, ctx)

        for i in eachindex(graph.adj)
            @test graph.directcount[i] <= neighbors_length(graph.adj, i)
        end
        @test any(graph.directcount[i] < neighbors_length(graph.adj, i) for i in eachindex(graph.adj))
    end
end

