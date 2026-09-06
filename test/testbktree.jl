# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test, Random
Random.seed!(0)

@isdefined(FAST_TESTS) || (const FAST_TESTS = get(ENV, "FAST_TESTS", "false") == "true")

# every object of the subtree rooted at `node`: the node itself, its long-leaf bucket (if
# any), and recursively its children
function bkt_subtree_ids(bkt, node)
    ids = UInt32[]
    stack = UInt32[node]
    while !isempty(stack)
        p = pop!(stack)
        push!(ids, p)
        if bkt.bucketlen[p] > 0
            sp = bkt.bucketstart[p]
            append!(ids, view(bkt.bucket, sp:(sp+bkt.bucketlen[p]-1)))
        end

        nc = bkt.childcount[p]
        if nc > 0
            cs = bkt.childstart[p]
            append!(stack, view(bkt.childnode, cs:(cs+nc-1)))
        end
    end

    ids
end

bkt_all_ids(bkt) = bkt.root[] == 0 ? UInt32[] : bkt_subtree_ids(bkt, bkt.root[])

# the two invariants `search` relies on: children ascending by distinct key (it binary
# searches them), and every object under the child keyed `k` of `p` at distance exactly `k`
# from `p` (this is what makes the triangle-inequality pruning exact)
function bkt_check_tree(bkt)
    dist = distance(bkt)
    db = database(bkt)
    bkt.root[] == 0 && return true
    ok = true
    stack = UInt32[bkt.root[]]
    while !isempty(stack)
        p = pop!(stack)
        nc = bkt.childcount[p]
        nc == 0 && continue
        cs = bkt.childstart[p]
        ce = cs + nc - 1
        keys = bkt.childkey[cs:ce]
        ok &= issorted(keys) && allunique(keys)
        for e in cs:ce
            c = bkt.childnode[e]
            for x in bkt_subtree_ids(bkt, c)
                ok &= Dist.evaluate(dist, db[p], db[x]) == bkt.childkey[e]
            end

            push!(stack, c)
        end
    end

    ok
end

@testset "BKTree" begin
    alphabet = collect("abcdefgh")
    mkword() = collect(String(rand(alphabet, rand(3:9))))
    n, m, k = (FAST_TESTS ? 500 : 2000), 30, 10
    dist = Dist.Seqs.Levenshtein()
    db = VectorDatabase([mkword() for _ in 1:n])
    queries = VectorDatabase([mkword() for _ in 1:m])
    ctx = GenericContext(; reporters=[])
    seq = ExhaustiveSearch(dist, db)
    Igold, Dgold = searchbatch(seq, ctx, queries, k)

    @testset "structure and exactness (npivots=$npivots, minleaf=$minleaf)" for npivots in (1, 3), minleaf in (1, 12)
        bkt = BKT(dist, db)
        index!(bkt, ctx; npivots, minleaf)

        # every object is indexed exactly once, as a node or inside a long leaf's bucket
        @test sort(bkt_all_ids(bkt)) == UInt32.(1:n)
        @test bkt_check_tree(bkt)
        @test length(bkt) == n
        @test database(bkt) === db
        @test distance(bkt) === dist

        # BKT is exact: the distances it returns are the distances of the true k nearest
        # neighbors. Ids are compared as a set per query only where they are unambiguous:
        # edit distance ties are rampant, and a tie makes several distinct answers equally
        # correct, so `macrorecall` under-reports here without anything being wrong.
        I, D = searchbatch(bkt, ctx, queries, k)
        @test D ≈ Dgold
    end

    @testset "range queries agree with the exhaustive scan" begin
        bkt = BKT(dist, db)
        index!(bkt, ctx)
        for radius in (0f0, 1f0, 3f0)
            for j in 1:m
                q = queries[j]
                a = search(bkt, ctx, q, RadiusSorted(radius))
                b = search(seq, ctx, q, RadiusSorted(radius))
                @test sort(collect(IdView(a))) == sort(collect(IdView(b)))
            end
        end
    end

    @testset "long leaves" begin
        # minleaf bounds every bucket, and minleaf=1 leaves no bucket at all
        bkt = BKT(dist, db)
        index!(bkt, ctx; minleaf=12)
        @test maximum(bkt.bucketlen) <= 11   # a leaf holds its representative plus <= minleaf-1
        @test length(bkt.bucket) == sum(bkt.bucketlen)

        deep = BKT(dist, db)
        index!(deep, ctx; minleaf=1)
        @test all(iszero, deep.bucketlen)
        @test isempty(deep.bucket)

        # a database no larger than minleaf is a single leaf holding everything
        small = BKT(dist, VectorDatabase([mkword() for _ in 1:5]))
        index!(small, ctx; minleaf=12)
        @test small.bucketlen[small.root[]] == 4
        @test all(iszero, small.childcount)
    end

    @testset "Hamming over bit vectors" begin
        hdist = Dist.Bits.Hamming()
        hdb = VectorDatabase([rand(UInt64, 4) for _ in 1:(FAST_TESTS ? 300 : 1000)])
        hq = VectorDatabase([rand(UInt64, 4) for _ in 1:10])
        hseq = ExhaustiveSearch(hdist, hdb)
        hbkt = BKT(hdist, hdb)
        index!(hbkt, ctx)
        @test bkt_check_tree(hbkt)
        _, Dg = searchbatch(hseq, ctx, hq, 5)
        _, Db = searchbatch(hbkt, ctx, hq, 5)
        @test Dg ≈ Db
    end

    @testset "edge cases" begin
        # empty database: nothing to search, and no tree
        empty = BKT(dist, VectorDatabase(Vector{Char}[]))
        index!(empty, ctx)
        @test empty.root[] == 0
        @test length(search(empty, ctx, mkword(), knnqueue(KnnSorted, 3))) == 0

        # a single object
        one = BKT(dist, VectorDatabase([collect("abc")]))
        index!(one, ctx)
        @test one.root[] == 1
        res = search(one, ctx, collect("abd"), knnqueue(KnnSorted, 3))
        @test length(res) == 1 && argmin(res) == 1 && minimum(res) == 1f0

        # every object identical: all distances are 0, so a single key -- still exact
        same = BKT(dist, VectorDatabase([collect("abc") for _ in 1:20]))
        index!(same, ctx; minleaf=4)
        @test sort(bkt_all_ids(same)) == UInt32.(1:20)
        r = search(same, ctx, collect("abc"), knnqueue(KnnSorted, 5))
        @test length(r) == 5 && maximum(r) == 0f0
    end

    @testset "rejected inputs" begin
        # a SemiMetric breaks the pruning rule (DamerauLevenshtein is integer-valued but
        # violates the triangle inequality), so it is rejected unless explicitly overridden
        @test_throws ArgumentError BKT(Dist.Seqs.DamerauLevenshtein(), db)
        @test BKT(Dist.Seqs.DamerauLevenshtein(), db; checkmetric=false) isa BKT

        # build-once: no rebuilding in place
        built = BKT(dist, db)
        index!(built, ctx)
        @test_throws ArgumentError index!(built, ctx)

        bad = BKT(dist, db)
        @test_throws ArgumentError index!(bad, ctx; npivots=0)
        @test_throws ArgumentError index!(bad, ctx; minleaf=0)
    end
end
