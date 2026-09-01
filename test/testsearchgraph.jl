using SimilaritySearch, Random, StatsBase, Statistics
using Test
#using AllocCheck

@isdefined(FAST_TESTS) || (const FAST_TESTS = get(ENV, "FAST_TESTS", "false") == "true")

#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function prepare_benchmark(Database;
    ksearch::Int=8,
    # kept comfortably above SearchGraphContext's default starting_callback=256 (and its
    # first logbase_callback=1.5 checkpoint at 384) so the hints callback still fires more
    # than once -- smaller n hit an unrelated empty-hints edge case in matrixhints.
    n::Int=(FAST_TESTS ? 800 : 2_000),
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

@testset "RadiusSorted/RadiusHeap via searchbatch! with SearchGraph" begin
    # n is kept < 64 so `search(bs::BeamSearch, index::SearchGraph, ...)` takes its
    # brute-force branch (every item is evaluated unconditionally), guaranteeing an exact
    # match against a brute-force radius scan -- this isolates RadiusSorted/RadiusHeap's own
    # push_item! admission logic from BeamSearch's approximate neighborhood exploration.
    dim, n, m = 4, 50, 5
    dist = Dist.SqL2()
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))

    graph = SearchGraph(dist, db)
    ctx = SearchGraphContext(neighborhood=Neighborhood(filter=SatNeighborhood()), verbose=false)
    index!(graph, ctx)

    alldists = [Dist.evaluate(dist, queries[j], db[i]) for i in 1:n, j in 1:m]
    radius = Float32(quantile(vec(alldists), 0.3))

    for QueueType in (RadiusSorted, RadiusHeap)
        knns = [QueueType(radius) for _ in 1:m]
        searchbatch!(graph, ctx, queries, knns)

        for j in 1:m
            gold = sort(IdDist[IdDist(i, alldists[i, j]) for i in 1:n if alldists[i, j] <= radius], by=x -> x.dist)
            got = collect(IdDistView(knns[j]))
            @test length(got) == length(gold)
            @test Set(x.id for x in got) == Set(x.id for x in gold)
            @test all(x.dist <= radius for x in got)
        end
    end
end

@testset "IdentityNeighborhood passes candidates through instead of producing empty neighborhoods" begin
    # Regression test for issue #58: `neighborhoodfilter(::IdentityNeighborhood, ...)` used to
    # return its result instead of writing into `output`, and `find_neighborhood!` only ever
    # reads `output` -- so every node silently ended up with zero neighbors. Nothing in the
    # suite exercised this filter, which is why it went unnoticed.
    dim, n, m, ksearch = 8, 2_000, 30, 8
    dist = Dist.SqL2()
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))

    graph = SearchGraph(dist, db)
    ctx = SearchGraphContext(neighborhood=Neighborhood(filter=IdentityNeighborhood()), verbose=false)
    index!(graph, ctx)

    @test all(>(0), neighbors_length.(Ref(graph.adj), 1:length(graph)))

    seq = ExhaustiveSearch(dist, db)
    ectx = GenericContext()
    gold_ids, _ = searchbatch(seq, ectx, queries, ksearch)
    knns_ids, _ = searchbatch(graph, ctx, queries, ksearch)
    @test macrorecall(gold_ids, knns_ids) >= 0.8
end

@testset "rebuild resets a tiny inherited maxvisits instead of inheriting it" begin
    # Regression test for issue #59: rebuild's own neighbor search (and the auto-tuning
    # execute_callbacks! triggers afterward, which is itself anchored to whatever algo[] it's
    # handed) used to inherit g.algo[] verbatim. A `maxvisits` tuned down for a smaller/partial
    # graph or a different, cheap proxy distance would then silently cap every node's
    # rebuild-time search, baking a permanently degraded topology into the result. `bsize`/`Δ`
    # (the fields optimize_index! actually explores) should still carry over; only `maxvisits`
    # gets reset.
    dim, n = 8, 500
    dist = Dist.SqL2()
    db = MatrixDatabase(rand(Float32, dim, n))

    graph = SearchGraph(dist, db)
    index!(graph, SearchGraphContext(verbose=false))
    graph.algo[] = BeamSearch(; bsize=graph.algo[].bsize, Δ=graph.algo[].Δ, maxvisits=1)

    G = rebuild(graph, SearchGraphContext(hyperparameters_callback=nothing); progress=nothing)

    @test G.algo[].maxvisits == BeamSearch().maxvisits
    @test G.algo[].bsize == graph.algo[].bsize
    @test G.algo[].Δ == graph.algo[].Δ
    @test graph.algo[].maxvisits == 1  # rebuild must not mutate its input
end

@testset "MaxMatchError doesn't blow up on degenerate (zero-spread) gold neighborhoods" begin
    # Regression test: matcherror's ρ(q) used to add only eps(Float32) as a floor over the
    # gold neighborhood's own spread, so a fully degenerate query (its k gold neighbors all
    # tied at the same distance -- routine with duplicate points, or near-duplicate items on
    # real data) made ρ(q) collapse to ≈eps(Float32); dividing by that inflated an ordinary,
    # non-buggy distance mismatch (here: 0.001, well within normal floating-point/approximate-
    # search noise) by a factor of ~10^6-10^7, letting a single such query dominate a whole
    # batch's mean MatchError. `minspread` now floors ρ(q) at something meaningful instead.
    ctx = GenericContext()
    res = knnqueue(ctx, 3)
    push_item!(res, 1, 1.001f0)
    push_item!(res, 2, 1.001f0)
    push_item!(res, 3, 1.001f0)
    golddist = Float32[1.0, 1.0, 1.0]  # fully tied -- true spread is exactly 0

    @test SimilaritySearch.matcherror(golddist, res, 1f0, 1f0, 0f0) > 100      # old behavior: blows up
    @test SimilaritySearch.matcherror(golddist, res, 1f0, 1f0, 1f-2) < 1       # fixed: bounded, sane
end

@testset "index!(...; :bitsketch)" begin
    dim, n, m, ksearch = 64, 2_000, 30, 8
    dist = Dist.SqL2()
    db = MatrixDatabase(randn(Float32, dim, n))
    queries = MatrixDatabase(randn(Float32, dim, m))
    ctx = SearchGraphContext(verbose=false)

    graph = SearchGraph(dist, db)
    index!(graph, ctx, :bitsketch)
    @test length(graph) == n
    @test all(>(0), neighbors_length.(Ref(graph.adj), 1:n))
    # algo[] must stay untouched (issue #59's bug: carrying over the sketch-space-tuned
    # BeamSearch would miscalibrate every later search/optimize call against the real dist).
    @test graph.algo[] == BeamSearch()

    optimize_index!(graph, ctx, MinRecall(0.9))
    seq = ExhaustiveSearch(dist, db)
    ectx = GenericContext()
    gold_ids, _ = searchbatch(seq, ectx, queries, ksearch)
    knns_ids, _ = searchbatch(graph, ctx, queries, ksearch)
    @test macrorecall(gold_ids, knns_ids) >= 0.7

    # :qr requires nbits <= dim (an orthogonal rotation can't grow dimensionality)
    graph_qr = SearchGraph(dist, db)
    index!(graph_qr, ctx, :bitsketch; method=:qr, nbits=64)
    @test length(graph_qr) == n

    @test_throws ArgumentError index!(SearchGraph(dist, db), ctx, :bitsketch; nbits=100)  # not a multiple of 64
    @test_throws ArgumentError index!(graph, ctx, :bitsketch)  # graph is no longer empty
    @test_throws ArgumentError index!(SearchGraph(dist, VectorDatabase([rand(Float32, dim) for _ in 1:n])), ctx, :bitsketch)  # not a MatrixDatabase
end
