# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, SimilaritySearch.RaBitQ, LinearAlgebra, Random, Statistics
using SimilaritySearch: evaluate, encode, encodequery, rotate
using SimilaritySearch.Projections: RandomizedHadamard
const SQ = SimilaritySearch.ScalarQuant

unitvectors(rng, dim, n) = (X = randn(rng, Float32, dim, n); foreach(j -> normalize!(view(X, :, j)), 1:n); X)
knn_ids(index, ctx, queries, k) = [Int32.(collect(IdView(search(index, ctx, q, knnqueue(KnnSorted, k))))) for q in queries]
recall_of(gold, got) = mean(length(intersect(g, r)) / length(g) for (g, r) in zip(gold, got))
qrrotation(rng, dim) = SimilaritySearch.Projections.qr(rng, Float32, dim, dim)
truecos(X, i, q) = Float32(dot(view(X, :, i), q))

@testset "encoding: the rotations preserve the norm and the code carries what the estimate needs" begin
    rng = Xoshiro(1)
    for (dim, rot) in ((384, qrrotation(rng, 384)), (512, RandomizedHadamard(512; rng)))
        est = RaBitQCosine(rot)
        @test est.dim == dim
        o = randn(rng, Float32, dim)
        r = rotate(est.rot, o)
        @test norm(r) ≈ norm(o) rtol=1f-4                               # an orthogonal rotation
        @test norm(rotate(est.rot, o .+ 1f0)) ≈ norm(o .+ 1f0) rtol=1f-4
        code = encode(est, o)
        @test length(code.bits) == cld(dim, 64)
        @test 0 < code.c <= 1
        @test code.norm ≈ norm(o) rtol=1f-4
        @test code.err > 0
        # a bit is set exactly where the rotated coordinate is non-negative
        @test count(>=(0), r) == sum(count_ones, code.bits)
        q = encodequery(est, o)
        @test norm(q.r) ≈ 1 rtol=1f-4
        @test q.norm ≈ norm(o) rtol=1f-4
        # the object against its own code: the estimate is the exact cosine, 1, up to Float32
        @test abs(estimatecos(est, q, code) - 1) <= 1f-3
        @test_throws MethodError RaBitQCosine(nothing)                    # the bits need the rotation
        @test_throws ArgumentError RaBitQCosine(SimilaritySearch.Projections.qr(rng, Float32, dim, 32))   # a projection, not a rotation
        @test_throws ArgumentError encode(est, randn(Float32, dim + 1))
        @test_throws ArgumentError encodequery(est, randn(Float32, dim - 1))
    end
    @test_throws ArgumentError RandomizedHadamard(384)                  # not a power of two
    # a constant vector, which the plain Walsh-Hadamard transform would collapse onto one
    # coordinate, spreads over the coordinates once the signs are randomized
    est = RaBitQCosine(RandomizedHadamard(256; rng))
    @test count(!=(0), rotate(est.rot, fill(1f0, 256))) > 100
end

@testset "the signed-sum kernel matches a scalar reference at every word boundary" begin
    # the SIMD kernel expands whole 64-bit words and leaves a partial last word to a scalar
    # tail; both halves, and dimensions that fall on either side of a word boundary, against
    # the plain loop
    rng = Xoshiro(9)
    scalar(bits, r, m) = m * sum(((bits[((i - 1) >>> 6) + 1] >>> ((i - 1) & 63)) & 1 == 1 ? r[i] : -r[i]) for i in eachindex(r))
    for dim in (1, 7, 63, 64, 65, 100, 127, 128, 129, 200, 384, 500)
        est = RaBitQCosine(qrrotation(rng, dim))
        o = randn(rng, Float32, dim)
        code = encode(est, o)
        q = encodequery(est, randn(rng, Float32, dim))
        @test abs(RaBitQ._signeddot(code.bits, q.r, est.m) - scalar(code.bits, q.r, est.m)) <= 1f-4
        @test abs(estimatecos(est, encodequery(est, o), code) - 1) <= 1f-3
    end
end

@testset "the estimate is unbiased and its error bound covers it" begin
    rng = Xoshiro(2)
    dim, n = 384, 2000
    est = RaBitQCosine(qrrotation(rng, dim))
    X = unitvectors(rng, dim, n)
    codes = rabitqcodes(est, X)
    @test length(codes) == n
    @test codes isa VectorDatabase
    q = normalize!(randn(rng, Float32, dim))
    qq = encodequery(est, q)
    errs = [estimatecos(est, qq, codes[i]) - truecos(X, i, q) for i in 1:n]
    @test abs(mean(errs)) < 0.01                                        # unbiased
    # random unit vectors in 384-d have true cosines of spread 1/sqrt(384) ≈ 0.05, against an
    # estimate error of about 0.04, so the correlation sits near 0.8 here; on data with
    # neighbors the spread is larger and so is the correlation
    @test cor([estimatecos(est, qq, codes[i]) for i in 1:n], [truecos(X, i, q) for i in 1:n]) > 0.7
    # the bound is a confidence interval at ε₀ = 1.9, whose nominal coverage is
    # 1 - exp(-ε₀² / 2) ≈ 0.84; measured here at about 0.80
    covered = count(i -> abs(errs[i]) <= errorbound(est, codes[i]), 1:n) / n
    @test 0.7 < covered < 0.95
    # dissimilarity, and the same estimate reached through the distance
    @test all(i -> evaluate(est, qq, codes[i]) ≈ 1 - estimatecos(est, qq, codes[i]), 1:20)
    @test all(i -> evaluate(est, codes[i], qq) == evaluate(est, qq, codes[i]), 1:20)
end

"Points around `nc` random centers: distances with a spread the estimate's error does not drown, unlike i.i.d. Gaussians."
function clustered(rng, dim, nc, per; spread=0.5f0)
    C = randn(rng, Float32, dim, nc)
    hcat((C[:, j] .+ spread .* randn(rng, Float32, dim) for j in 1:nc for _ in 1:per)...)
end

@testset "the Euclidean estimator, and the estimate between two codes" begin
    rng = Xoshiro(3)
    dim = 128
    est = RaBitQL2(qrrotation(rng, dim))
    X = clustered(rng, dim, 10, 100)
    n = size(X, 2)
    codes = rabitqcodes(est, X)
    q = X[:, 1] .+ 0.3f0 .* randn(rng, Float32, dim)
    qq = encodequery(est, q)
    trued = [Float32(norm(view(X, :, i) .- q)) for i in 1:n]
    estd = [evaluate(est, qq, codes[i]) for i in 1:n]
    @test cor(estd, trued) > 0.95
    @test abs(mean(estd .- trued)) / mean(trued) < 0.05
    # between two stored codes, the SimHash estimate: monotone in the true cosine
    cc = [evaluate(RaBitQCosine(est.rot, dim, est.m), codes[1], codes[i]) for i in 2:n]
    tc = [1 - dot(normalize(X[:, 1]), normalize(X[:, i])) for i in 2:n]
    @test cor(cc, tc) > 0.9
    @test evaluate(est, codes[5], codes[5]) == 0f0
    # and the Euclidean one between codes uses the stored norms
    @test evaluate(est, codes[5], codes[6]) > 0
end

@testset "an AsymmetricSearchGraph over RaBitQ codes takes raw items and raw queries" begin
    rng = Xoshiro(4)
    dim, n, nq, k = 128, 5000, 100, 10
    X = unitvectors(rng, dim, n)
    Q = unitvectors(rng, dim, nq)
    queries = collect(eachcol(Q))
    gold = knn_ids(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)   # unit vectors: L2 order == cosine order
    est = RaBitQCosine(qrrotation(rng, dim))

    # the estimator alone, exhaustively, with queries prepared by hand
    seq = ExhaustiveSearch(est, rabitqcodes(est, X))
    rseq = recall_of(gold, knn_ids(seq, GenericContext(), [encodequery(est, q) for q in queries], k))
    @test rseq > 0.2

    # the graph: raw in, raw queries, the graph rotates once per item and per query
    G = AsymmetricSearchGraph(est, rabitqcodes())
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    @test length(G) == n
    @test database(G)[3].bits == encode(est, X[:, 3]).bits
    rg = recall_of(gold, knn_ids(G, ctx, queries, k))
    @test rg > 0.2
    @test rg >= rseq - 0.15                                             # the graph against the estimator's own ceiling
    @test_throws ArgumentError index!(G, ctx)
    optimize_index!(G, ctx, MinRecall(0.9); queries=MatrixDatabase(Q))
    push_item!(G, ctx, X[:, 1])
    @test length(G) == n + 1 && database(G)[n + 1].bits == database(G)[1].bits
    @test sprint(show, est) == "RaBitQCosine(dim=128, rotation=RandomProjections)"
    @info "RaBitQ over $n unit vectors in $dim-d: exhaustive recall@$k = $(round(rseq; digits=3)), graph = $(round(rg; digits=3))"
end

@testset "RaBitQRefined: the bits navigate, the fine level re-evaluates inside the estimate" begin
    rng = Xoshiro(21)
    dim, n, nq, k = 128, 5000, 100, 10
    X = unitvectors(rng, dim, n)
    Q = unitvectors(rng, dim, nq)
    queries = collect(eachcol(Q))
    gold = knn_ids(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)
    coarse = RaBitQCosine(qrrotation(rng, dim))
    rbits = recall_of(gold, knn_ids(ExhaustiveSearch(coarse, rabitqcodes(coarse, X)), GenericContext(), [encodequery(coarse, q) for q in queries], k))

    # exact fine level, τ = Inf: every distance is the exact cosine, so exhaustively recall is 1
    ex = RaBitQRefined(coarse, RaBitQExactFallback())
    codes = rabitqcodes(ex, X)
    @test codes[1] isa Tuple && length(codes[1][2]) == dim && eltype(codes[1][2]) == Float32
    qs = [encodequery(ex, q) for q in queries]
    @test recall_of(gold, knn_ids(ExhaustiveSearch(ex, codes), GenericContext(), qs, k)) > 0.99
    @test abs(evaluate(ex, qs[1], codes[7]) - (1 - dot(Q[:, 1], X[:, 7]))) <= 1f-4
    @test evaluate(ex, codes[7], qs[1]) == evaluate(ex, qs[1], codes[7])
    # Float16 halves the fine level
    ex16 = RaBitQRefined(coarse, RaBitQExactFallback{Float16}())
    c16 = encode(ex16, X[:, 7])
    @test eltype(c16[2]) == Float16 && abs(evaluate(ex16, qs[1], c16) - (1 - dot(Q[:, 1], X[:, 7]))) <= 2f-3

    # a τ on the data's scale: only what the bits cannot rule out is re-evaluated, and what
    # the search keeps is within it, so the exhaustive recall stays
    τ = refinethreshold(coarse, X, k; rng)
    @test 0 < τ < 2
    exτ = RaBitQRefined(coarse, RaBitQExactFallback(); τ)
    @test recall_of(gold, knn_ids(ExhaustiveSearch(exτ, codes), GenericContext(), qs, k)) > 0.9
    # the farthest object under the bits keeps its bit estimate: its lower bound is beyond τ
    far = argmax(i -> evaluate(coarse, qs[1], codes[i][1]), 1:n)
    @test evaluate(exτ, qs[1], codes[far]) == evaluate(coarse, qs[1], codes[far][1])
    # between two stored objects only the bits are compared
    @test evaluate(exτ, codes[1], codes[2]) == evaluate(coarse, codes[1][1], codes[2][1])

    # a scalar-quantized fine level at 4 bits sits between the bits and the exact level
    f4 = RaBitQVectorFallback(SQ.SQgu4, coarse, X; rng)
    @test SQ.quantizer(f4) === SQ.SQgu4 && SQ.codewidth(f4) == 4 && SQ.isglobal(f4)
    @test sprint(show, f4) == "RaBitQVectorFallback(SQgu4, dim=$dim)"
    sq4 = RaBitQRefined(coarse, f4)
    codes4 = rabitqcodes(sq4, X)
    @test codes4[1][2] isa SQ.SQVec{4}
    r4 = recall_of(gold, knn_ids(ExhaustiveSearch(sq4, codes4), GenericContext(), qs, k))
    @test rbits < r4 < 1.0
    # the per-vector family needs no data, and at 8 bits it is close to the exact level
    f8 = RaBitQVectorFallback(SQ.SQu8, coarse)
    @test SQ.quantizer(f8) === SQ.SQu8 && !SQ.isglobal(f8) && f8.E === nothing
    sq8 = RaBitQRefined(coarse, f8)
    codes8 = rabitqcodes(sq8, X)
    @test codes8[1][2] isa SQ.SQVec{8}
    @test abs(evaluate(sq8, qs[1], codes8[7]) - (1 - dot(Q[:, 1], X[:, 7]))) <= 1f-2
    @test recall_of(gold, knn_ids(ExhaustiveSearch(sq8, codes8), GenericContext(), qs, k)) > r4
    @test RaBitQVectorFallback(SQ.SQgu8, coarse; minmax=(-0.3, 0.3)) isa RaBitQVectorFallback{8,SQ.SQMinC}
    @test_throws ArgumentError RaBitQVectorFallback(SQ, coarse, X; rng)                 # not a quantizer module
    @test_throws ArgumentError RaBitQVectorFallback(SQ.SQgu4, coarse)                    # a global range needs data or minmax

    # through the graph: raw items in, raw queries in, and the refined graph beats the bits' own ceiling
    @test eltype(rabitqcodes(exτ).vecs) == typeof(codes[1]) && eltype(rabitqcodes(coarse).vecs) == RaBitQCode{Vector{UInt64}}
    G = AsymmetricSearchGraph(exτ, rabitqcodes(exτ))
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    @test length(G) == n
    rg = recall_of(gold, knn_ids(G, ctx, queries, k))
    @test rg > rbits
    @test sprint(show, exτ) isa String
    @info "RaBitQRefined over $n unit vectors in $dim-d: bits alone $(round(rbits; digits=3)) exhaustive, exact fine level through the graph $(round(rg; digits=3)) at τ=$(round(τ; digits=3)), SQ4 fine level $(round(r4; digits=3)) exhaustive"
end
