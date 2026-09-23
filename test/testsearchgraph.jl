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

function abs_minrecall(B; filter=SatNeighborhood(), kwargs...)
    @info "===================== minrecall $(typeof(filter)) $kwargs =============================="
    graph = SearchGraph(B.dist, B.db; kwargs...)
    ctx = SearchGraphContext(
        neighborhood=Neighborhood(; filter),
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

@testset "radius search navigates when the ball starts out of reach (#67)" begin
    # The testset above keeps n < 64 so `search` takes its brute-force branch; this one is the
    # opposite case, the one that crashed: a graph large enough to be navigated, with a radius so
    # small that no entry point falls inside it. `RadiusSorted` rejected every hint, stayed empty,
    # and `nearest(res)` read past the end of a 0-element vector.
    dim, n = 4, 3000
    dist = Dist.SqL2()
    db = MatrixDatabase(rand(Xoshiro(1), Float32, dim, n))
    graph = SearchGraph(dist, db)
    ctx = SearchGraphContext(verbose=false)
    index!(graph, ctx)

    queries = [rand(Xoshiro(100 + j), Float32, dim) for j in 1:20]
    balls = [Set(i for i in 1:n if Dist.evaluate(dist, q, db[i]) <= r) for q in queries, r in (0.02f0,)]

    for QueueType in (RadiusSorted, RadiusHeap)
        found = 0
        wanted = 0
        for (j, q) in enumerate(queries)
            res = search(graph, ctx, q, QueueType(0.02f0))
            got = collect(IdDistView(res))
            # never returns anything outside the ball, and never the navigation reserve
            @test all(p -> p.dist <= 0.02f0, got)
            @test Set(p.id for p in got) ⊆ balls[j]
            found += length(got)
            wanted += length(balls[j])
        end

        # the real regression this guards: a guarded-but-ungradiented search returns *empty* balls
        @test wanted > 0
        @test found >= 0.6 * wanted
    end

    # a radius no object can satisfy must come back empty, not crash
    res = search(graph, ctx, queries[1], RadiusSorted(0f0))
    @test length(res) == 0

    # kmin is a keyword; every value keeps the answer a subset of the true ball
    for kmin in (1, 2, 8, 64)
        res = search(graph, ctx, queries[1], RadiusSorted(0.05f0); kmin)
        @test Set(p.id for p in IdDistView(res)) ⊆ Set(i for i in 1:n if Dist.evaluate(dist, queries[1], db[i]) <= 0.05f0)
    end
    @test_throws ArgumentError search(graph, ctx, queries[1], RadiusSorted(0.05f0); kmin=0)

    # searchbatch! reaches the same method
    Q = MatrixDatabase(hcat(queries...))
    knns = [RadiusSorted(0.05f0) for _ in queries]
    searchbatch!(graph, ctx, Q, knns)
    @test all(length(r) > 0 for r in knns)
end

@testset "optimize_index! tunes for a radius workload (#67)" begin
    Random.seed!(0xBA11)
    dim, n = 8, 2_000
    dist = Dist.SqL2()
    db = MatrixDatabase(randn(Float32, dim, n))
    graph = SearchGraph(dist, db)
    ctx = SearchGraphContext(verbose=false)
    index!(graph, ctx)

    queries = [randn(Float32, dim) for _ in 1:40]
    alldists = [sort!([Dist.evaluate(dist, q, db[i]) for i in 1:n]) for q in queries]
    radius = Float32(sum(d[5] for d in alldists) / length(queries))   # balls of ~5-10 members

    function ballrecall()
        found, wanted = 0, 0
        for (q, dd) in zip(queries, alldists)
            wanted += count(<=(radius), dd)
            found += length(search(graph, ctx, q, RadiusSorted(radius)))
        end
        found / wanted
    end

    # tuning against ball gold, which only MaxMatchError can score
    optimize_index!(graph, ctx, MaxMatchError(; maxerror=0.01f0); radius, kmin=8, numqueries=32)
    @test graph.algo[] isa BeamSearch
    @test ballrecall() >= 0.8

    # the recall-based goals cannot: macrorecall divides by the gold ball's size, and a small
    # radius routinely leaves a query with an empty ball
    @test_throws ArgumentError optimize_index!(graph, ctx, MinRecall(0.9); radius)
    @test_throws ArgumentError optimize_index!(graph, ctx, ParetoRecall(); radius)
end

@testset "matcherror scores the ball, not the navigation reserve" begin
    # Regression guard with teeth: SearchModels swallows exceptions raised while evaluating a
    # configuration ("ignoring configuration due to exception") and optimization still returns a
    # result, so a broken matcherror on BallKnn looks like a successful tuning run. It is asserted
    # here directly instead.
    golddist = Float32[0.1, 0.2, 0.3]           # the true ball: 3 members within radius 1.0

    perfect = SimilaritySearch.BallKnn(1.0f0, 2)
    for (i, d) in enumerate((0.1f0, 0.2f0, 0.3f0))
        push_item!(perfect, i, d)
    end
    @test SimilaritySearch.matcherror(golddist, perfect, 1f0, 1f0, 1f-2) == 0.0

    # same three ball members, plus reserve items *outside* the radius: the reserve must not
    # change the score, and must not be mistaken for ball members that were found
    withreserve = SimilaritySearch.BallKnn(1.0f0, 6)
    for (i, d) in enumerate((0.1f0, 0.2f0, 0.3f0, 5f0, 6f0, 7f0))
        push_item!(withreserve, i, d)
    end
    @test length(withreserve) > length(SimilaritySearch.ballview(withreserve))
    @test SimilaritySearch.matcherror(golddist, withreserve, 1f0, 1f0, 1f-2) == 0.0

    # a search that reached only one of the three ball members pays η for each one it missed
    partial = SimilaritySearch.BallKnn(1.0f0, 4)
    for (i, d) in enumerate((0.1f0, 4f0, 5f0, 6f0))
        push_item!(partial, i, d)
    end
    @test SimilaritySearch.ninside(partial) == 1
    @test SimilaritySearch.matcherror(golddist, partial, 1f0, 1f0, 1f-2) ≈ 2/3
    # an empty true ball is free: there was nothing to find
    @test SimilaritySearch.matcherror(Float32[], partial, 1f0, 1f0, 1f-2) == 0.0
end

@testset "BallKnn keeps a navigation reserve outside the ball" begin
    # the queue #67's fix navigates with: the ball plus at least `kmin` nearest items, whichever
    # is larger, so it is never empty and its `maximum` is a threshold that actually moves
    res = SimilaritySearch.BallKnn(1.0f0, 4)
    @test length(res) == 0
    @test maximum(res) == typemax(Float32)      # nothing to bound the search with yet

    for (i, d) in enumerate((9f0, 8f0, 7f0, 6f0, 5f0))
        push_item!(res, i, d)
    end
    @test length(res) == 4                      # trimmed to the reserve
    @test maximum(res) == 8f0                   # the k-th distance, shrinking
    @test SimilaritySearch.maxlength(res) == 4  # == capacity, what optimize_index!'s cov block reads
    @test length(res) == SimilaritySearch.maxlength(res)
    @test length(SimilaritySearch.ballview(res)) == 0   # nothing is inside the ball yet
    @test nearest(res).dist == 5f0

    for (i, d) in enumerate((0.5f0, 0.25f0, 0.75f0, 0.9f0, 0.1f0))
        push_item!(res, 100 + i, d)
    end
    @test length(SimilaritySearch.ballview(res)) == 5   # every item within the radius is kept
    @test maximum(res) == 1.0f0                          # flattened at the radius
    @test all(p -> p.dist <= 1.0f0, SimilaritySearch.ballview(res))

    # a sixth ball member grows the queue past kmin; the reserve no longer bounds it
    push_item!(res, 200, 0.3f0)
    @test length(SimilaritySearch.ballview(res)) == 6
    @test length(res) == 6

    reuse!(res)
    @test length(res) == 0
    @test maximum(res) == typemax(Float32)
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
    # seeded: the recall assertions below are statistical, and every testset that runs earlier in
    # this file consumes the global RNG (graph construction samples hints and neighborhoods), so
    # without this the data here shifts whenever a testset is added above and a borderline
    # threshold starts failing for reasons that have nothing to do with :bitsketch
    Random.seed!(0xB175)
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

    # width > 1 bootstraps through QuantSketch's m-bit codes instead of sign bits. The model
    # is NOT resized by it: all four graphs below are built from the same 512 hyperplanes,
    # and `width` only changes how many bits each hyperplane's value is kept with (so the
    # sketch grows to 512*width bits). That is the point -- more precision on the same,
    # already-fitted model.
    for width in (2, 4, 8)
        g = SearchGraph(dist, db)
        index!(g, ctx, :bitsketch; method=:gaussian, nbits=512, width)
        @test length(g) == n
        @test all(>(0), neighbors_length.(Ref(g.adj), 1:n))
        @test g.algo[] == BeamSearch()
        optimize_index!(g, ctx, MinRecall(0.9))
        knns_ids, _ = searchbatch(g, ctx, queries, ksearch)
        @test macrorecall(gold_ids, knns_ids) >= 0.7
    end

    @test_throws ArgumentError index!(SearchGraph(dist, db), ctx, :bitsketch; nbits=100)  # not a multiple of 64
    @test_throws ArgumentError index!(SearchGraph(dist, db), ctx, :bitsketch; width=3)    # not 1, 2, 4 or 8
    # a precomputed sketch carries no quantization range, so it cannot be widened
    @test_throws ArgumentError index!(SearchGraph(dist, db), ctx, :bitsketch;
                                      method=:external, nbits=512, width=2,
                                      sketch=rand(UInt64, 8, n))
    @test_throws ArgumentError index!(graph, ctx, :bitsketch)  # graph is no longer empty
    @test_throws ArgumentError index!(SearchGraph(dist, VectorDatabase([rand(Float32, dim) for _ in 1:n])), ctx, :bitsketch)  # not a MatrixDatabase
    @test_throws ArgumentError index!(SearchGraph(dist, VectorDatabase([rand(Float32, dim) for _ in 1:n])), ctx, :bitsketch; nbits=512, width=2)
end

@testset "every NeighborhoodFilter builds a usable graph" begin
    # KCentersNeighborhood shipped unusable (#64: it crashed on the second insertion, where the
    # candidate set has exactly one item) precisely because nothing here ever built with it.
    B = prepare_benchmark(MatrixDatabase)
    for filter in (DistalSatNeighborhood(), KCentersNeighborhood())
        graph, ctx = abs_minrecall(B; filter)
        @test all(neighbors_length(graph.adj, i) > 0 for i in eachindex(graph.adj))
    end
end

@testset "find_neighborhood! resolves the degenerate candidate sets itself" begin
    # `find_neighborhood!` guarantees filters at least two candidates: zero happens on the first
    # insertion, one on the second. KCentersNeighborhood is the filter that derives a size from
    # the candidate count, so it is the one that notices when that guarantee breaks.
    dist = Dist.SqL2()
    ctx = SearchGraphContext(neighborhood=Neighborhood(filter=KCentersNeighborhood()), verbose=false)

    for n in 1:4   # the sizes where the candidate set is degenerate or barely not
        graph = SearchGraph(dist, MatrixDatabase(rand(Float32, 4, n)))
        index!(graph, ctx)
        @test length(graph) == n
        # the second insertion is the one handed a single candidate: it must still connect
        n >= 2 && @test neighbors_length(graph.adj, 2) > 0
    end

    # and the filter itself survives a single candidate (defense in depth: `log2(m + 1)` never
    # asks fft for zero centers). It is driven as `find_neighborhood!` drives it, with
    # `sortitems!(tmp)`, not a raw queue.
    graph = SearchGraph(dist, MatrixDatabase(rand(Float32, 4, 300)))
    index!(graph, ctx)
    res, out = knnqueue(ctx, 4), knnqueue(ctx, 4)
    push_item!(res, 2, 0.5f0)
    @test length(SimilaritySearch.neighborhoodfilter(KCentersNeighborhood(), graph, ctx, database(graph, 1), sortitems!(res), out)) == 1
    push_item!(res, 3, 0.7f0)
    @test length(SimilaritySearch.neighborhoodfilter(KCentersNeighborhood(), graph, ctx, database(graph, 1), sortitems!(res), reuse!(out))) == 2
end
