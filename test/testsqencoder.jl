# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, LinearAlgebra, Random, Statistics
using SimilaritySearch: evaluate, encode, encodequery
using SimilaritySearch.ScalarQuant: SQEncoder, sqcodes
using SimilaritySearch.Projections: RandomizedHadamard
const SQ = SimilaritySearch.ScalarQuant

unitvectors(rng, dim, n) = (X = randn(rng, Float32, dim, n); foreach(j -> normalize!(view(X, :, j)), 1:n); X)
knn_ids(index, ctx, queries, k) = [Int32.(collect(IdView(search(index, ctx, q, knnqueue(KnnSorted, k))))) for q in queries]
recall_of(gold, got) = mean(length(intersect(g, r)) / length(g) for (g, r) in zip(gold, got))
qrrotation(rng, dim) = SimilaritySearch.Projections.qr(rng, Float32, dim, dim)

@testset "SQEncoder: rotate once, quantize, evaluate with ScalarQuant's kernels" begin
    rng = Xoshiro(31)
    dim, n, nq, k = 64, 3000, 100, 10
    X = unitvectors(rng, dim, n)
    Q = unitvectors(rng, dim, nq)
    queries = collect(eachcol(Q))
    gold = knn_ids(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)

    # the quantizer module names the family and the width; the rotation is the object, or nothing
    for (quant, rot, bits, global_) in ((SQ.SQgu8, qrrotation(rng, dim), 8, true), (SQ.SQgu4, RandomizedHadamard(dim; rng), 4, true),
                                        (SQ.SQu2, qrrotation(rng, dim), 2, false), (SQ.SQu8, nothing, 8, false),
                                        (SQ.SQgu2, nothing, 2, true))
        e = SQEncoder(quant, rot, X; rng)
        @test SQ.quantizer(e) === quant
        @test SQ.codewidth(e) == bits && SQ.isglobal(e) == global_
        @test e.dim == dim
        o = X[:, 5]
        c = encode(e, o)
        @test c isa SQ.SQVec{bits} && length(c) == dim
        qr = encodequery(e, o)
        @test qr isa SQ.SQQuery{bits} && qr isa AbstractVector{Float32}   # prepared, and still the rotated query (#110)
        @test norm(qr) ≈ norm(o) rtol=1f-4
        # the object against its own code: within the quantization step of a unit vector, plus
        # the few 1e-6 the query's 15-bit image adds at 8 bits
        @test evaluate(e, qr, c) <= (bits == 8 ? 2f-4 : bits == 4 ? 2f-2 : 0.3f0)
        @test evaluate(e, c, qr) == evaluate(e, qr, c)
        @test evaluate(e, c, c) == 0f0                                    # codes against codes: exact zero
        # a rotation preserves distances: the estimate is the exact SqL2 up to quantization
        c7 = encode(e, X[:, 7])
        truth = sum(abs2, o .- X[:, 7])
        @test abs(evaluate(e, qr, c7) - truth) <= (bits == 8 ? 1f-2 : bits == 4 ? 0.1f0 : 0.6f0) * max(1f0, truth)
        @test_throws ArgumentError encode(e, randn(Float32, dim + 1))
    end
    @test_throws ArgumentError SQEncoder(SQ, nothing, X)                                   # not a quantizer module
    @test_throws ArgumentError SQEncoder(SQ.SQgu8, nothing, dim)                           # a global range needs data or minmax
    @test_throws ArgumentError SQEncoder(SQ.SQgu8, qrrotation(rng, 32), X)                 # rotation and data disagree
    @test_throws ArgumentError SQEncoder(SQ.SQu8, SimilaritySearch.Projections.qr(rng, Float32, dim, 32), dim)   # a projection, not a rotation
    @test SQEncoder(SQ.SQgu8, nothing, dim; minmax=(-0.5, 0.5)) isa SQEncoder{8,Nothing,SQ.SQMinC}
    # without a rotation argument nothing is rotated: the default
    @test SQEncoder(SQ.SQgu8, X) isa SQEncoder{8,Nothing,SQ.SQMinC} && SQEncoder(SQ.SQu4, dim) isa SQEncoder{4,Nothing,SQ.AutoRange}   # a per-vector encoder carries its range policy
    @test encode(SQEncoder(SQ.SQgu8, dim; minmax=(-1, 1)), X[:, 1]).V == encode(SQEncoder(SQ.SQgu8, nothing, dim; minmax=(-1, 1)), X[:, 1]).V
    @test SQEncoder(SQ.SQu4, RandomizedHadamard(dim; rng), dim) isa SQEncoder{4,RandomizedHadamard,SQ.AutoRange}
    @test SQEncoder(SQ.SQu4, dim; range=SQ.ExtremaRange()) isa SQEncoder{4,Nothing,SQ.ExtremaRange}
    @test sprint(show, SQEncoder(SQ.SQgu4, nothing, X)) == "SQEncoder(SQgu4, nothing, dim=$dim, dist=SimilaritySearch.ScalarQuant.SqL2())"
    @test sprint(show, SQEncoder(SQ.SQu2, qrrotation(rng, dim), dim)) == "SQEncoder(SQu2, RandomProjections, dim=$dim, dist=SimilaritySearch.ScalarQuant.SqL2())"
    # without a rotation and with the same range, the codes are ScalarQuant's own
    e0 = SQEncoder(SQ.SQgu8, nothing, dim; minmax=extrema(X))
    db0 = SQ.GlobalQuantDatabase(8, X; minmax=extrema(X))
    @test encode(e0, X[:, 3]).V == db0[3].V && encode(e0, X[:, 3]).E == db0[3].E

    # the storage is a QuantDatabase with the estimator's parameters, in dense blocks, and it
    # takes the codes as they are
    e = SQEncoder(SQ.SQgu8, qrrotation(rng, dim), X; rng)
    db = sqcodes(e, X)
    @test db isa SQ.QuantDatabase{8} && length(db) == n && db.Q isa BlockMatrixDatabase
    @test db[3].V == encode(e, X[:, 3]).V && db[3].E == e.E
    @test length(sqcodes(e)) == 0

    # through the graph: raw in, raw queries, and the same recall as the unrotated 8-bit
    # asymmetric graph up to what one build's tuning moves; the codes are the encoder's
    G = AsymmetricSearchGraph(e, sqcodes(e))
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    @test length(G) == n
    @test database(G)[11].V == encode(e, X[:, 11]).V
    rrot = recall_of(gold, knn_ids(G, ctx, queries, k))
    @test rrot > 0.4
    # a rotation changes nothing an 8-bit quantizer can see: the estimator's own exhaustive
    # recall matches the unrotated one (the graphs' recalls depend on the beam each tuning
    # leaves, so they are not what is compared)
    rex = recall_of(gold, knn_ids(ExhaustiveSearch(e, db), GenericContext(), [encodequery(e, q) for q in queries], k))
    rex8 = recall_of(gold, knn_ids(ExhaustiveSearch(SQ.SqL2(), SQ.GlobalQuantDatabase(8, X; minmax=extrema(X))), GenericContext(), queries, k))
    @test abs(rex - rex8) <= 0.03 && rex > 0.9
    optimize_index!(G, ctx, MinRecall(0.9); queries=MatrixDatabase(Q))
    push_item!(G, ctx, X[:, 2])
    @test length(G) == n + 1 && database(G)[n + 1].V == database(G)[2].V
    # the per-vector family and a cosine through the same path
    ep = SQEncoder(SQ.SQu4, RandomizedHadamard(dim; rng), dim; dist=SQ.Cosine())
    Gp = AsymmetricSearchGraph(ep, sqcodes(ep))
    append_items!(Gp, ctx, MatrixDatabase(X))
    @test recall_of(gold, knn_ids(Gp, ctx, queries, k)) > 0.3
    @info "SQEncoder over $n unit vectors in $dim-d: exhaustive rotated $(round(rex; digits=3)) vs unrotated $(round(rex8; digits=3)); graph $(round(rrot; digits=3))"
end
