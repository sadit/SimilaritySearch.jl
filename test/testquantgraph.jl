# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, LinearAlgebra, Random, Statistics
import Distances
import Distances: evaluate

const SQ = SimilaritySearch.ScalarQuant

"Ids of the `k` nearest neighbors of every query, in order, through the ordinary search interface."
knn_ids(index, ctx, queries, k) =
    [Int32.(collect(IdView(search(index, ctx, q, knnqueue(KnnSorted, k))))) for q in queries]

recall_of(gold, got) = mean(length(intersect(g, r)) / length(g) for (g, r) in zip(gold, got))

inner(G::SearchGraph) = G
inner(G::AsymmetricSearchGraph) = G.graph

# Two estimators, at top level because `struct` cannot sit inside a testset. Both are plain
# types with their parameters as fields: they serialize with the graph, nothing is a closure.

"A distance that encodes its own 8-bit global codes for a plain storage and evaluates a raw query against them."
struct GlobalCodes <: AbstractEstimator
    E::SQ.SQMinC
    dim::Int
end
SimilaritySearch.encode(e::GlobalCodes, v) = SQ.SQVec{8}(e.E, SQ.packcodes!(Val(8), Vector{UInt8}(undef, e.dim), v, e.E.min, 1f0 / e.E.c))
Distances.evaluate(::GlobalCodes, q, stored) = evaluate(SQ.SqL2(), q, stored)

"A distance that stores a 2-bit code next to an 8-bit one and re-evaluates from the fine one when the coarse estimate is below `τ`."
struct TwoLevel <: AbstractEstimator
    coarse::SQ.GlobalQuantDatabase{2,MatrixDatabase{Matrix{UInt8}}}   # the parameters; empty, used to quantize
    fine::SQ.GlobalQuantDatabase{8,MatrixDatabase{Matrix{UInt8}}}
    τ::Float32
end
SimilaritySearch.encode(e::TwoLevel, v) = (SQ.quantize(e.coarse, v), SQ.quantize(e.fine, v))
function Distances.evaluate(e::TwoLevel, q, stored)
    d = evaluate(SQ.SqL2(), q, stored[1])
    d > e.τ ? d : evaluate(SQ.SqL2(), q, stored[2])
end
# the neighborhood filters compare a new item's candidates among themselves: stored against stored
Distances.evaluate(::TwoLevel, a::Tuple, b::Tuple) = evaluate(SQ.SqL2(), a[2], b[2])

@testset "symmetric and asymmetric graphs over quantized storage (#86)" begin
    # A graph stored as codes works in one of two ways, fixed when it is built: a SearchGraph
    # over the codes inserts and searches with them (symmetric), an AsymmetricSearchGraph over
    # the same storage inserts and searches with the raw objects against the codes. What is
    # asserted is the ordering the policies must respect, never a recall value: for a fixed
    # graph, full-precision queries do at least as well as quantized ones, and the same
    # topology searched in full precision at least as well as either. Widths: both ends of
    # the range, where quantization error dominates (2) and nearly vanishes (8), and the full
    # cross product once, in the middle (4).
    Random.seed!(7)
    dim, n, nq, k = 32, 3000, 100, 10
    X = randn(Float32, dim, n); foreach(j -> normalize!(view(X, :, j)), 1:n)
    Q = randn(Float32, dim, nq); foreach(j -> normalize!(view(Q, :, j)), 1:nq)
    queries = collect(eachcol(Q))
    mm = extrema(X)
    gold = knn_ids(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)
    slack = 0.03      # what 100 queries over 3000 points resolve

    function build(bits, mode)
        db = SQ.GlobalQuantDatabase(bits, BlockMatrixDatabase(dim ÷ SQ.codesperbyte(Val(bits)), UInt8), mm; dim)
        G = mode == :symmetric ? SearchGraph(SQ.SqL2(), db) : AsymmetricSearchGraph(SQ.SqL2(), db)
        ctx = SearchGraphContext(; reporters=[])
        append_items!(G, ctx, MatrixDatabase(X))
        G, ctx, db
    end

    function policies(G, ctx, db)
        rfull = recall_of(gold, knn_ids(G, ctx, queries, k))                                        # Float32 queries
        rcodes = recall_of(gold, knn_ids(inner(G), ctx, [SQ.quantize(db, q) for q in queries], k))  # quantized queries
        g = inner(G)   # the same edges and hints, searched over the full-precision vectors
        Gx = SearchGraph(Dist.SqL2(), MatrixDatabase(X); adj=g.adj, hints=g.hints, algo=g.algo, len=g.len)
        rexact = recall_of(gold, knn_ids(Gx, ctx, queries, k))
        rfull, rcodes, rexact
    end

    for bits in (2, 8), mode in (:symmetric, :asymmetric)
        G, ctx, db = build(bits, mode)
        @test G isa AbstractSearchGraph
        @test length(G) == n
        @test database(G) === db
        @test db == SQ.GlobalQuantDatabase(bits, X; minmax=mm)      # the storage is codes either way
        rfull, rcodes, rexact = policies(G, ctx, db)
        @test rfull >= rcodes - slack
        @test rexact >= max(rfull, rcodes) - slack
        # a sanity floor only: with the default BeamSearch and no optimize_index!, 32-d
        # Gaussian data lands around 0.65-0.75 here, and the value is not what is asserted
        @test rexact > 0.4
        bits == 8 && @test rfull > 0.4
    end

    # the middle width, every combination
    results = Dict{Tuple{Symbol,Symbol},Float64}()
    for mode in (:symmetric, :asymmetric)
        G, ctx, db = build(4, mode)
        rfull, rcodes, rexact = policies(G, ctx, db)
        results[(mode, :full)] = rfull
        results[(mode, :codes)] = rcodes
        @test rfull >= rcodes - slack
        @test rexact >= max(rfull, rcodes) - slack
    end
    @test all(>(0.4), values(results))

    # the asymmetric graph has no raw object to work with on its own
    G, ctx, db = build(8, :asymmetric)
    @test_throws ArgumentError index!(G, ctx)
    @test_throws ArgumentError rebuild(G, ctx)
    @test_throws ArgumentError optimize_index!(G, ctx, MinRecall(0.9))
    optimize_index!(G, ctx, MinRecall(0.9); queries=MatrixDatabase(Q))     # raw queries: fine
    @test length(G) == n
    push_item!(G, ctx, X[:, 1])
    @test length(G) == n + 1 && collect(database(G)[n + 1].V) == collect(db[1].V)
    @test sprint(show, G) isa String
end

@testset "an asymmetric graph over a database that stores what it is given is the symmetric graph" begin
    # Sequential insertion (`parallel_block=1`), no hyperparameters callback and the same seed
    # before each build: the parallel path appends reverse links in whatever order the threads
    # finish, and the tuning callback draws random queries (from the stored items in one graph
    # and from the raw ones in the other), so two builds are otherwise not the same graph.
    Random.seed!(11)
    dim, n = 16, 1500
    X = randn(Float32, dim, n)
    graphs = map((SearchGraph, AsymmetricSearchGraph)) do T
        Random.seed!(11)
        G = T(Dist.SqL2(), BlockMatrixDatabase(dim, Float32))
        ctx = SearchGraphContext(; reporters=[], parallel_block=1, hyperparameters_callback=nothing)
        append_items!(G, ctx, MatrixDatabase(X))
        G
    end
    @test length(graphs[1]) == length(graphs[2]) == n
    @test all(i -> collect(neighbors(graphs[1].adj, i)) == collect(neighbors(graphs[2].graph.adj, i)), 1:n)
end

@testset "Cosine against a plain vector agrees with the dequantized cosine, and drives an asymmetric graph" begin
    Random.seed!(3)
    dim, n = 64, 500
    X = randn(Float32, dim, n)
    db = SQ.GlobalQuantDatabase(8, X; minmax=extrema(X))
    q = randn(Float32, dim)
    for i in (1, 7, n)
        a = Float64[db[i][t] for t in 1:dim]
        truth = 1.0 - dot(a, q) / (norm(a) * norm(q))
        @test abs(evaluate(SQ.Cosine(), db[i], q) - truth) <= 1f-4
        @test evaluate(SQ.Cosine(), q, db[i]) == evaluate(SQ.Cosine(), db[i], q)
        # the query's scale does not matter, as a cosine's must not
        @test abs(evaluate(SQ.Cosine(), db[i], 3f0 .* q) - evaluate(SQ.Cosine(), db[i], q)) <= 1f-5
    end
    @test evaluate(SQ.Cosine(), db[1], zeros(Float32, dim)) == 1f0

    gdb = SQ.GlobalQuantDatabase(8, BlockMatrixDatabase(dim, UInt8), extrema(X); dim)
    G = AsymmetricSearchGraph(SQ.Cosine(), gdb)
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    res = search(G, ctx, X[:, 5], knnqueue(KnnSorted, 3))
    @test nearest(res).id == 5
end

@testset "an estimator is the distance: it encodes for the storage and evaluates raw against stored" begin
    # The sketch path: the storage is a plain growable database of codes, the estimator's
    # `encode` produces them, and its `evaluate` takes a raw query against a code. Its
    # parameters are fields, so the graph and the estimator serialize together.
    Random.seed!(5)
    dim, n = 32, 400
    X = randn(Float32, dim, n)
    mm = extrema(X)
    est = GlobalCodes(SQ.SQMinC(Float32(mm[1]), 1f0 / SQ.sqglobalscale(255, mm[1], mm[2])), dim)
    @test est isa Distances.PreMetric && isbits(est)
    codes = VectorDatabase(type=typeof(SimilaritySearch.encode(est, X[:, 1])))
    G = AsymmetricSearchGraph(est, codes)
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    @test length(G) == n
    @test database(G)[3].V == SimilaritySearch.encode(est, X[:, 3]).V
    res = search(G, ctx, X[:, 9], knnqueue(KnnSorted, 3))
    @test nearest(res).id == 9
    # a plain distance encodes nothing: the storage is what transforms
    v = X[:, 1]
    @test SimilaritySearch.encode(SQ.SqL2(), v) === v
end

@testset "an estimator re-evaluates inside its evaluation, transparently to the graph" begin
    # The scalar quantizers need no correction: their distance is the answer. An estimator
    # with an error keeps what it needs beside the code and re-evaluates inside `evaluate`
    # when its estimate cannot be trusted; the graph only ever evaluates the distance. This
    # one stores a 2-bit code next to an 8-bit one and refines from the fine code whenever
    # the coarse estimate is below a threshold -- at 2 bits, where the coarse kernel misranks
    # the most, that has to recover recall the coarse estimate alone cannot.
    Random.seed!(13)
    dim, n, nq, k = 32, 3000, 100, 10
    X = randn(Float32, dim, n); foreach(j -> normalize!(view(X, :, j)), 1:n)
    Q = randn(Float32, dim, nq); foreach(j -> normalize!(view(Q, :, j)), 1:nq)
    queries = collect(eachcol(Q))
    mm = extrema(X)
    gold = knn_ids(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)

    empty2 = SQ.GlobalQuantDatabase(2, Matrix{UInt8}(undef, dim ÷ 4, 0), mm; dim)   # parameters only
    empty8 = SQ.GlobalQuantDatabase(8, Matrix{UInt8}(undef, dim, 0), mm; dim)
    ctx = SearchGraphContext(; reporters=[])       # each graph keeps the parameters its own tuning leaves

    # coarse only: the 2-bit graph
    coarse = AsymmetricSearchGraph(SQ.SqL2(), SQ.GlobalQuantDatabase(2, BlockMatrixDatabase(dim ÷ 4, UInt8), mm; dim))
    append_items!(coarse, ctx, MatrixDatabase(X))
    rcoarse = recall_of(gold, knn_ids(coarse, ctx, queries, k))

    # two levels, refining every estimate below τ. Unit vectors: squared distances in [0, 4],
    # and the 2-bit estimate carries a positive bias of about 0.7 here (32 coordinates of
    # quantization noise), so τ sits above it: what is not clearly far gets refined
    est = TwoLevel(empty2, empty8, 2.5f0)
    stored = VectorDatabase(type=typeof(SimilaritySearch.encode(est, X[:, 1])))
    two = AsymmetricSearchGraph(est, stored)
    append_items!(two, ctx, MatrixDatabase(X))
    @test length(two) == n
    @test database(two)[5][1].V == SQ.quantize(empty2, X[:, 5]).V && database(two)[5][2].V == SQ.quantize(empty8, X[:, 5]).V
    rtwo = recall_of(gold, knn_ids(two, ctx, queries, k))
    @test rtwo >= rcoarse + 0.05
    # what comes out is the estimator's own evaluation, in order
    res = search(two, ctx, queries[1], knnqueue(KnnSorted, k))
    for p in IdDistView(res)
        @test p.dist == evaluate(est, queries[1], database(two)[p.id])
    end
    @test issorted([p.dist for p in IdDistView(res)])
    # a radius queue goes through the same distance
    ball = search(two, ctx, queries[3], RadiusSorted(0.9f0))
    @test ball isa RadiusSorted
end

"An estimator whose query side is a rotation: `encodequery` rotates the raw query once, `encode` stores the rotated object at 8 bits, `evaluate` is the mixed kernel in the rotated space."
struct Rotated8 <: AbstractEstimator
    R::Matrix{Float32}                                            # orthogonal
    params::SQ.GlobalQuantDatabase{8,MatrixDatabase{Matrix{UInt8}}}
end
SimilaritySearch.encodequery(e::Rotated8, q) = e.R' * q
SimilaritySearch.encode(e::Rotated8, o) = SQ.quantize(e.params, e.R' * o)
Distances.evaluate(::Rotated8, q, stored) = evaluate(SQ.SqL2(), q, stored)

@testset "encodequery prepares the raw query once, on every path that takes one" begin
    Random.seed!(17)
    dim, n, nq, k = 32, 2000, 60, 10
    X = randn(Float32, dim, n); foreach(j -> normalize!(view(X, :, j)), 1:n)
    Q = randn(Float32, dim, nq); foreach(j -> normalize!(view(Q, :, j)), 1:nq)
    queries = collect(eachcol(Q))
    gold = knn_ids(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)
    R = Matrix(qr(randn(Float32, dim, dim)).Q)
    XR = R' * X
    est = Rotated8(R, SQ.GlobalQuantDatabase(8, Matrix{UInt8}(undef, dim, 0), extrema(XR); dim))
    # the default is the identity
    v = X[:, 1]
    @test SimilaritySearch.encodequery(SQ.SqL2(), v) === v

    G = AsymmetricSearchGraph(est, VectorDatabase(type=typeof(SimilaritySearch.encode(est, v))))
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))                       # raw items in; rotated codes stored
    @test length(G) == n
    @test database(G)[7].V == SQ.quantize(est.params, XR[:, 7]).V
    rrot = recall_of(gold, knn_ids(G, ctx, queries, k))            # raw queries in; rotated once inside
    @test rrot > 0.4
    # the same data through an unrotated 8-bit asymmetric graph: a rotation preserves L2, so the
    # two must agree up to what one build's tuning moves
    G8 = AsymmetricSearchGraph(SQ.SqL2(), SQ.GlobalQuantDatabase(8, BlockMatrixDatabase(dim, UInt8), extrema(X); dim))
    append_items!(G8, ctx, MatrixDatabase(X))
    r8 = recall_of(gold, knn_ids(G8, ctx, queries, k))
    @test abs(rrot - r8) <= 0.1
    # raw queries are prepared on the other two paths as well
    optimize_index!(G, ctx, MinRecall(0.9); queries=MatrixDatabase(Q))
    @test recall_of(gold, knn_ids(G, ctx, queries, k)) > 0.4
    push_item!(G, ctx, X[:, 3])
    @test length(G) == n + 1 && database(G)[n + 1].V == database(G)[3].V
    # a query prepared by hand and passed to the inner graph is what the wrapper evaluates
    res1 = collect(IdView(search(G, ctx, queries[1], knnqueue(KnnSorted, k))))
    res2 = collect(IdView(search(G.graph, ctx, R' * queries[1], knnqueue(KnnSorted, k))))
    @test res1 == res2
end
