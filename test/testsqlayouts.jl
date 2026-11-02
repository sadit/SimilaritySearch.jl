# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, LinearAlgebra, Random
using SimilaritySearch: evaluate, encode
using SimilaritySearch.ScalarQuant: SQEncoder, sqcodes, GlobalCodesDatabase, SQCodes
const SQ = SimilaritySearch.ScalarQuant

# The two layouts of a quantized database hold the same codes and must give the same
# distances, to the bit: `GlobalCodesDatabase` drops the code sums, recomputed when a distance
# needs them.
@testset "quantized layouts: same codes, same distances" begin
    rng = Xoshiro(41)
    dim, n = 64, 2000
    X = randn(rng, Float32, dim, n)
    foreach(j -> normalize!(view(X, :, j)), 1:n)
    q = normalize!(randn(rng, Float32, dim))
    dists = (SQ.SqL2(), SQ.L2(), SQ.L1(), SQ.NormCosine(), SQ.Cosine())
    for bits in (8, 4, 2)
        g = SQ.GlobalQuantDatabase(bits, X)
        c = GlobalCodesDatabase(g)
        @test c[1] isa SQCodes{bits}
        @test SQ.SQVec(c[5]).Sa == g[5].Sa && SQ.SQVec(c[5]).Saa == g[5].Saa
        Q = SQ.SQQuery{bits}(q)
        for d in dists, i in 1:40
            for j in (1, 17, 999)
                r = evaluate(d, g[i], g[j])
                @test evaluate(d, c[i], c[j]) === r
                @test evaluate(d, c[i], g[j]) === r && evaluate(d, g[i], c[j]) === r
            end
            @test evaluate(d, c[i], q) === evaluate(d, g[i], q)
            @test evaluate(d, c[i], Q) === evaluate(d, g[i], Q)
        end
        # identical codes are at exactly zero in every layout (the equal-scale branch)
        @test evaluate(SQ.SqL2(), c[3], c[3]) === 0f0 && evaluate(SQ.L1(), c[3], g[3]) === 0f0

        # codes taken back with their range
        E2 = GlobalCodesDatabase(bits, c.Q, (g.E.min, g.E.min + SQ.levels(Val(bits)) * g.E.c)).E
        @test E2.min == g.E.min && E2.c ≈ g.E.c
    end
    @test_throws ArgumentError GlobalCodesDatabase(4, MatrixDatabase(zeros(UInt8, 10, 3)), (0, 1); dim=63)
end

@testset "quantized layouts: growth through the encoder" begin
    rng = Xoshiro(42)
    dim, n = 48, 3000
    X = randn(rng, Float32, dim, n)
    for quant in (SQ.SQgu8, SQ.SQgu4, SQ.SQu8, SQ.SQu2)
        enc = SQEncoder(quant, X; rng)
        ref = sqcodes(enc, X)
        for layout in (:codes,)
            SQ.isglobal(enc) || (@test_throws ArgumentError sqcodes(enc; layout); continue)
            a = sqcodes(enc, X; layout)            # all threads, straight into the blocks
            b = sqcodes(enc; layout)
            for i in 1:n
                push_item!(b, encode(enc, view(X, :, i)))
            end
            @test length(a) == length(b) == n
            @test all(i -> a[i].V == ref[i].V == b[i].V && a[i].E == ref[i].E == b[i].E, 1:n)
            @test all(i -> evaluate(SQ.SqL2(), a[i], a[1]) === evaluate(SQ.SqL2(), ref[i], ref[1]), 1:n)
            sc = SimilaritySearch.spreadcopy(a)
            @test typeof(sc) == typeof(a) && sc == a
        end
    end
    enc = SQEncoder(SQ.SQgu8, X; rng)
    other = SQEncoder(SQ.SQgu8, 2X; rng)
    @test_throws ArgumentError push_item!(sqcodes(enc; layout=:codes), encode(other, X[:, 1]))
    @test_throws ArgumentError sqcodes(enc; layout=:rows)
end

@testset "quantized layouts: one graph, both layouts" begin
    rng = Xoshiro(43)
    dim, n = 32, 4000
    X = randn(rng, Float32, dim, n)
    enc = SQEncoder(SQ.SQgu8, X; rng)
    G = SearchGraph(SQ.SqL2(), sqcodes(enc, X))
    ctx = SearchGraphContext(hyperparameters_callback=OptimizeParameters(MinRecall(0.9)))
    index!(G, ctx)
    qs = VectorDatabase([encode(enc, view(X, :, i)) for i in 1:300])
    ids, ds = searchbatch(G, ctx, qs, 10)
    H = SearchGraph(G.dist, sqcodes(enc, X; layout=:codes), G.adj, G.hints, Ref(G.algo[]), Ref(G.len[]), G.members)
    ids2, ds2 = searchbatch(H, ctx, qs, 10)
    @test ids2 == ids && ds2 == ds
end
