# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, Random, LinearAlgebra
using SimilaritySearch: reuse!, visited, visit!, check_visited_and_visit!, mayforget, newvisited

@testset "visited sets: semantics" begin
    rng = Xoshiro(7)
    n = 100_000
    for proto in (BitVisited(), ByteVisited(), HashVisited(; capacity=16), LossyHashVisited(; capacity=2^12))
        v = reuse!(newvisited(proto), n)
        ids = unique(rand(rng, 1:n, 3000))
        lossy = mayforget(v)
        for i in ids
            @test !check_visited_and_visit!(v, i)   # first time: not visited
        end
        # never a vertex that was not reached
        others = setdiff(rand(rng, 1:n, 3000), ids)
        @test !any(i -> visited(v, i), others)
        if lossy
            # 3000 ids in 4096 slots: most remain, some are forgotten
            kept = count(i -> visited(v, i), ids)
            @test 0.5length(ids) < kept < length(ids)
        else
            @test all(i -> visited(v, i), ids)
            @test all(i -> check_visited_and_visit!(v, i), ids)
        end
        # a new search sees nothing of the previous one, however similar
        reuse!(v, n)
        @test !any(i -> visited(v, i), ids)
        visit!(v, ids[1])
        @test visited(v, ids[1])
    end

    # the exact table grows with the visit and keeps its size for the next searches
    h = reuse!(HashVisited(; capacity=16), n)
    for i in 1:1000
        visit!(h, i)
    end
    @test length(h.slots) >= 2000 && all(i -> visited(h, i), 1:1000) && !visited(h, 1001)
    sz = length(h.slots)
    reuse!(h, n)
    @test length(h.slots) == sz && !visited(h, 1)

    # the byte table wraps every 255 searches without carrying anything over
    b = ByteVisited()
    for _ in 1:600
        reuse!(b, 1000)
        @test !visited(b, 7)
        visit!(b, 7)
    end
    @test b.gen != 0x00

    # generation wrap: the table is cleared and nothing survives
    for proto in (HashVisited(), LossyHashVisited())
        v = reuse!(newvisited(proto), n)
        visit!(v, 5)
        v.gen = 0xffffffff
        visit!(v, 6)
        reuse!(v, n)
        @test v.gen == 1 && !visited(v, 5) && !visited(v, 6)
    end
end

@testset "visited sets: search" begin
    rng = Xoshiro(11)
    dim, n, m, k = 8, 20_000, 200, 10
    X = rand(rng, Float32, dim, n)
    Q = rand(rng, Float32, dim, m)
    db, queries = MatrixDatabase(X), MatrixDatabase(Q)
    G = SearchGraph(Dist.SqL2(), db)
    ctx = SearchGraphContext(; reporters=[])
    index!(G, ctx)
    gold, _ = searchbatch(ExhaustiveSearch(Dist.SqL2(), db), GenericContext(), queries, k)
    run(c) = searchbatch(G, c, queries, k)
    I0, D0 = run(ctx)
    # the exact sets reach the same vertices in the same order: same answers, same cost
    for proto in (BitVisited(), ByteVisited(), HashVisited())
        c = SearchGraphContext(ctx; visited=proto)
        @test eltype(c.vstates) == typeof(proto)
        I, D = run(c)
        @test I == I0 && D == D0
    end
    # the lossy set: no repeated ids in any answer, recall close to the exact one (a 512-slot
    # table forces forgetting on a 20K-vertex graph)
    c = SearchGraphContext(ctx; visited=LossyHashVisited(; capacity=512))
    I, D = run(c)
    @test all(j -> allunique(view(I, :, j)), 1:m)
    r0 = macrorecall(gold, I0)
    @test macrorecall(gold, I) >= r0 - 0.05
end
