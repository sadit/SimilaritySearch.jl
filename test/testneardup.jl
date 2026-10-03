# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch, Random, Statistics
using SimilaritySearch: evaluate

"Ids of the `k` nearest neighbors of every query, through the ordinary search interface."
knn_ids(index, ctx, queries, k) = [Int32.(collect(IdView(search(index, ctx, q, knnqueue(KnnSorted, k))))) for q in queries]
recall_of(gold, got) = mean(length(intersect(g, r)) / length(g) for (g, r) in zip(gold, got))

@testset "near duplicates: members instead of nodes, expand and expand! give the raw neighbors back" begin
    rng = Xoshiro(7)
    dim, nbase, k = 8, 1500, 10
    B = rand(rng, Float32, dim, nbase)
    cols = [B[:, j] for j in 1:nbase]
    for j in 1:300, _ in 1:rand(rng, 1:20)       # 300 of the base points, copied 1-20 times: exact duplicates
        push!(cols, B[:, j])
    end
    shuffle!(rng, cols)
    X = MatrixDatabase(reduce(hcat, cols))
    n = length(X)
    ndup = n - nbase
    dist = Dist.SqL2()
    Q = MatrixDatabase(hcat(B[:, 1:50], rand(rng, Float32, dim, 50)))   # half the queries sit on a duplicated point
    queries = collect(eachcol(Q.matrix))
    E = ExhaustiveSearch(dist, X)
    goldI, goldD = searchbatch(E, GenericContext(), Q, k)
    gold = [Set(goldI[:, j]) for j in 1:length(Q)]

    @testset "the default is untouched: no members" begin
        G = SearchGraph(dist, X); ctx = SearchGraphContext(; reporters=[]); index!(G, ctx)
        @test isempty(G.members) && length(G) == n
        @test !ismember(G, 1) && representative(G, 1) == 1 && isempty(members(G, 1))
        res = search(G, ctx, Q[1], knnqueue(KnnSorted, k))
        before = collect(IdDistView(res))
        @test collect(expand(G, Q[1], res)) == before                        # nothing to add
        @test expand!(G, Q[1], res) === res && collect(IdDistView(res)) == before
        @test sprint(show, G.members) == "Members(0 members in 0 clusters)"
    end

    for parallel_block in (1, 64)
        @testset "neardup=0, parallel_block=$parallel_block" begin
            G = SearchGraph(dist, X)
            ctx = SearchGraphContext(; reporters=[], parallel_block, neighborhood=Neighborhood(; neardup=0f0))
            index!(G, ctx)
            @test length(G) == n                                                # members count
            nm = length(G.members)
            @test 0.95ndup <= nm <= ndup                                        # one node per cluster, up to what the search missed
            memberset = Set(keys(G.members.representative))
            for (m, r) in G.members.representative
                @test collect(neighbors(G.adj, m)) == [r]                       # a member: the single edge to its representative
                @test !ismember(G, r) && m in members(G, r)                     # which is a node, and lists it
                @test X[m] == X[r]                                              # and is an exact duplicate
            end
            @test all(i -> ismember(G, i) || all(v -> !(v in memberset), neighbors(G.adj, i)), 1:n)   # nothing links to a member
            @test sum(length(members(G, r)) for r in keys(G.members.lists)) == nm

            # the first stage answers with representatives: no member, at most one item per cluster
            res = search(G, ctx, Q[1], knnqueue(KnnSorted, k))
            @test !any(id -> ismember(G, id), IdView(res))
            @test allunique(representative(G, id) for id in IdView(res))

            # the second stage: expand (an iterator, nothing modified) and expand! (in place, trimmed to k)
            before = collect(IdDistView(res))
            ex = collect(expand(G, Q[1], res))
            @test collect(IdDistView(res)) == before
            @test length(ex) == length(before) + sum(length(members(G, p.id)) for p in before)
            @test all(p -> p.dist == evaluate(dist, Q[1], X[p.id]), ex)        # members carry their own evaluated distance
            @test issubset(Set(p.id for p in before), Set(p.id for p in ex))
            expand!(G, Q[1], res)
            @test length(res) == k && issorted(collect(DistView(res)))
            @test Set(IdView(res)) == Set(p.id for p in sort(ex; by=p -> p.dist)[1:k]) || length(unique(p.dist for p in ex)) < length(ex)   # the k nearest of the expansion (ties aside)

            # recall against the exhaustive gold: the first stage is capped on the duplicated queries, expand! lifts it
            r1 = recall_of(gold, knn_ids(G, ctx, queries, k))
            knns, dists = searchbatch(G, ctx, Q, k)
            @test all(j -> !any(id -> ismember(G, id), knns[:, j]), 1:length(Q))
            expand!(G, Q, knns, dists)
            r2 = recall_of(gold, [knns[:, j] for j in 1:length(Q)])
            @test r2 > r1 && r2 > 0.85
            @test all(j -> issorted(dists[:, j]), 1:length(Q))
            # the single-column form agrees with the matrix form
            ids1, d1 = searchbatch(G, ctx, MatrixDatabase(Q.matrix[:, 1:1]), k)
            expand!(G, Q[1], view(ids1, :, 1), view(d1, :, 1))
            @test ids1[:, 1] == knns[:, 1]

            # radius queue: expand! keeps what falls within the radius
            rad = search(G, ctx, Q[1], RadiusSorted(0.05f0))
            nrad = length(rad)
            expand!(G, Q[1], rad)
            @test length(rad) >= nrad && all(d -> d <= 0.05f0, DistView(rad))

            # tuning masks the whole cluster of an internal query and scores the expanded results
            optimize_index!(G, ctx, MinRecall(0.9); numqueries=32)
            @test G.algo[] isa BeamSearch
            optimize_index!(G, ctx, MaxMatchError(; maxerror=0.05f0); queries=Q)
            @test G.algo[] isa BeamSearch

            # rebuild keeps the members
            R = rebuild(G, ctx)
            @test length(R) == n && 0.95ndup <= length(R.members) <= ndup
            @test all(m -> collect(neighbors(R.adj, m)) == [representative(R, m)], keys(R.members.representative))
            knnsR, distsR = searchbatch(R, ctx, Q, k)
            expand!(R, Q, knnsR, distsR)
            @test recall_of(gold, [knnsR[:, j] for j in 1:length(Q)]) > 0.85
            @test occursin("members", sprint(show, G))
        end
    end

    @testset "near duplicates with ϵ > 0: members at a small distance, re-evaluated on expansion" begin
        base = rand(rng, Float32, dim, 500)
        cols = [base[:, j] for j in 1:500]
        for j in 1:100, _ in 1:3
            push!(cols, base[:, j] .+ 1f-3 .* randn(rng, Float32, dim))     # within SqL2 ≈ 8e-6 of the base point
        end
        shuffle!(rng, cols)
        Y = MatrixDatabase(reduce(hcat, cols))
        G = SearchGraph(dist, Y)
        ctx = SearchGraphContext(; reporters=[], neighborhood=Neighborhood(; neardup=1f-4))
        index!(G, ctx)
        @test 250 <= length(G.members) <= 300
        for (m, r) in G.members.representative
            @test 0 < evaluate(dist, Y[m], Y[r]) <= 1f-4 || evaluate(dist, Y[m], Y[r]) <= 2f-4   # chained near duplicates may sit a little farther
        end
        q = Y[1]
        res = search(G, ctx, q, knnqueue(KnnSorted, 5))
        ex = collect(expand(G, q, res))
        @test all(p -> p.dist == evaluate(dist, q, Y[p.id]), ex)
        @test length(unique(p.dist for p in ex)) > 1 || length(ex) == length(res)   # distances differ among a cluster
    end
end
