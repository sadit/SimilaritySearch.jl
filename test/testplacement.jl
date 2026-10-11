# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, Random
using SimilaritySearch: defaultblockbits, _staticadjlist, SPREAD_MIN
using SimilaritySearch.ScalarQuant
const SQ = SimilaritySearch.ScalarQuant

# Page spreading (1.6.5) changes where large stores lie, never what they hold: every check compares
# against the serial path or the source data. With one thread the spread paths are skipped.
@testset "page spread: block storage" begin
    @test page_spread()
    @test defaultblockbits(384, UInt8) == 17 && defaultblockbits(384, Float32) == 15   # >= 32 MB blocks
    rng = Xoshiro(3)
    n = 2SPREAD_MIN + 123
    X = rand(rng, Float32, 16, n)
    # bulk into an empty database: blocks full except the last, which holds exactly what is left
    A = BlockMatrixDatabase(16, Float32, 10)
    append_items!(A, eachcol(X))
    @test length(A) == n && all(i -> A[i] == view(X, :, i), 1:n)
    @test all(m -> size(m, 2) == 1024, A.blocks[1:end-1]) && size(A.blocks[end], 2) == n - 1024 * (length(A.blocks) - 1)
    # then single pushes (the last block grows) and a second bulk append (it fills up first)
    for i in 1:100
        push_item!(A, view(X, :, i))
    end
    append_items!(A, MatrixDatabase(X))
    @test length(A) == 2n + 100
    @test all(i -> A[n + i] == view(X, :, i), 1:100) && all(i -> A[n + 100 + i] == view(X, :, i), 1:n)
    @test all(m -> size(m, 2) == 1024, A.blocks[1:end-1])
    # pushing one by one from empty: a block starts at 256 columns and doubles
    B = BlockMatrixDatabase(16, Float32, 12)
    for i in 1:5000
        push_item!(B, view(X, :, i))
    end
    @test all(i -> B[i] == view(X, :, i), 1:5000) && [size(m, 2) for m in B.blocks] == [4096, 1024]
    # a failing item leaves the database as it was
    C = BlockMatrixDatabase(16, Float32, 10)
    append_items!(C, eachcol(X))
    bad = [k == n ÷ 2 ? rand(Float32, 3) : X[:, k] for k in 1:n]
    @test_throws Exception append_items!(C, bad)
    @test length(C) == n && all(i -> C[i] == view(X, :, i), 1:n)
    # off: the 1.6.4 layout
    set_page_spread!(false)
    try
        @test defaultblockbits(384, UInt8) == 8 && BlockMatrixDatabase(16, Float32) isa BlockMatrixDatabase{16,Float32,8}
        D = BlockMatrixDatabase(16, Float32)
        append_items!(D, eachcol(X))
        @test all(i -> D[i] == view(X, :, i), 1:n)
    finally
        set_page_spread!(true)
    end
end

@testset "page spread: quantized databases and graphs" begin
    rng = Xoshiro(5)
    dim, n = 32, SPREAD_MIN + 1000
    X = randn(rng, Float32, dim, n)
    for enc in (SQEncoder(SQ.SQgu8, X), SQEncoder(SQ.SQu4, X))
        d1 = sqcodes(enc, X)
        set_page_spread!(false)
        d2 = try sqcodes(enc, X) finally set_page_spread!(true) end
        @test length(d1) == length(d2) == n && all(i -> d1.Q[i] == d2.Q[i], 1:n)
        @test d1.Sa == d2.Sa && d1.Saa == d2.Saa && d1.E == d2.E
        # an asymmetric graph stores the same codes either way
        G = AsymmetricSearchGraph(enc, sqcodes(enc))
        append_items!(G, SearchGraphContext(; reporters=[]), MatrixDatabase(X))
        db = database(G)
        @test length(db) == n && all(i -> db.Q[i] == d2.Q[i], 1:n) && db.Sa == d2.Sa && db.E == d2.E
    end
    # plain vectors into a global database, and a wrongly quantized one rejected without a trace
    mm = extrema(X)
    g1 = SQ.GlobalQuantDatabase(8, BlockMatrixDatabase(dim, UInt8), mm; dim)
    append_items!(g1, MatrixDatabase(X))
    g2 = SQ.GlobalQuantDatabase(8, BlockMatrixDatabase(dim, UInt8, 8), mm; dim)
    for i in 1:n
        push_item!(g2, view(X, :, i))
    end
    @test all(i -> g1.Q[i] == g2.Q[i], 1:n) && g1.Sa == g2.Sa && g1.Saa == g2.Saa
    other = SQ.GlobalQuantDatabase(8, BlockMatrixDatabase(dim, UInt8), (mm[1] - 1f0, mm[2] + 1f0); dim)
    foreign = [SQ.quantize(other, view(X, :, i)) for i in 1:n]
    @test_throws Exception append_items!(g1, foreign)
    @test length(g1) == n && length(g1.Sa) == n && length(g1.Saa) == n
end

@testset "page spread: StaticAdjList" begin
    rng = Xoshiro(9)
    n = 3SPREAD_MIN
    adj = AdjList([UInt32.(rand(rng, 1:n, rand(rng, 0:30))) for _ in 1:n])
    a, b = StaticAdjList(adj), _staticadjlist(adj)
    @test a.offset == b.offset && a.end_point == b.end_point
    @test all(i -> neighbors(a, i) == neighbors(adj, i), 1:n)
end

@testset "spreadcopy" begin
    rng = Xoshiro(13)
    dim, n = 16, SPREAD_MIN + 500
    X = rand(rng, Float32, dim, n)
    @test spreadcopy(X) == X && spreadcopy(X) !== X
    M = spreadcopy(MatrixDatabase(X))
    @test M isa MatrixDatabase && M.matrix == X
    # a 1.6.4-style database (256-column blocks) comes back with huge-page blocks, same items
    set_page_spread!(false)
    B = try BlockMatrixDatabase(X) finally set_page_spread!(true) end
    C = spreadcopy(B)
    @test C isa BlockMatrixDatabase{dim,Float32,defaultblockbits(dim, Float32)} && length(C) == n && all(i -> C[i] == B[i], 1:n)
    for enc in (SQEncoder(SQ.SQgu8, X), SQEncoder(SQ.SQu4, X))
        d = sqcodes(enc, X)
        e = spreadcopy(d)
        @test length(e) == n && all(i -> e.Q[i] == d.Q[i], 1:n) && e.Sa == d.Sa && e.Saa == d.Saa && e.E == d.E
        @test e.Sa !== d.Sa
    end
    # graphs: the copy answers exactly as the original, dynamic and static adjacency alike
    G = SearchGraph(Dist.SqL2(), MatrixDatabase(X))
    ctx = SearchGraphContext(; reporters=[])
    index!(G, ctx)
    Q = MatrixDatabase(rand(rng, Float32, dim, 100))
    I0, D0 = searchbatch(G, ctx, Q, 10)
    G2 = spreadcopy(G)
    @test G2.adj isa AdjList && all(i -> neighbors(G2.adj, i) == neighbors(G.adj, i), 1:n)
    @test searchbatch(G2, ctx, Q, 10) == (I0, D0)
    S = SearchGraph(G.dist, G.db, StaticAdjList(G.adj), G.hints, G.algo, G.len, G.members)
    S2 = spreadcopy(S)
    @test S2.adj isa StaticAdjList && S2.adj.end_point == S.adj.end_point && searchbatch(S2, ctx, Q, 10) == (I0, D0)
    enc = SQEncoder(SQ.SQgu8, X)
    A = AsymmetricSearchGraph(enc, sqcodes(enc))
    append_items!(A, ctx, MatrixDatabase(X))
    A2 = spreadcopy(A)
    @test A2 isa AsymmetricSearchGraph && searchbatch(A2, ctx, Q, 10) == searchbatch(A, ctx, Q, 10)
end
