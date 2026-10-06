# Near-duplicate members on ccnews: the graph built with Neighborhood(neardup=0f0) against the default,
# scored on 10,500 held-out queries: first stage (representatives) and after expand! (raw neighbors).
# neardup=0f0 is floored at NEARDUP_NUMERICAL_ZERO (1e-5) by the package, see issue #99.
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> members-impact.jl
using SimilaritySearch, HDF5, Statistics, Random, Dates, Printf
const K = 10
path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}
Qi = h5read(path, "itest/queries"); Gi = h5read(path, "itest/knns")[1:K, :]
dim, n = size(X); db = MatrixDatabase(X)
eval_i = MatrixDatabase(Qi[:, 501:end]); gold_i = Gi[:, 501:end]
_, goldD = searchbatch(ExhaustiveSearch(Dist.SqL2(), db), GenericContext(), eval_i, K)
tied = findall(j -> goldD[K, j] - goldD[1, j] == 0, 1:size(goldD, 2))
@printf("held-out %d, fully tied gold %d\n", length(eval_i), length(tied))

function report(G, ctx, name)
    degs = [neighbors_length(G.adj, i) for i in 1:length(G)]
    @printf("%s: edges %d, max degree %d, members %d (%.1f%%), degree-1 nodes %d\n", name, sum(degs), maximum(degs), length(G.members), 100length(G.members) / length(G), count(==(1), degs))
    for (bsize, Δ) in ((8, 1.0), (16, 1.2))
        G.algo[] = BeamSearch(; bsize, Δ)
        before = copy(ctx.costdists)
        ids, dists = searchbatch(G, ctx, eval_i, K)
        v1 = distance_evaluations(ctx, before) / length(eval_i)
        pq1 = perqueryscores(recallscore, gold_i, ids)
        before = copy(ctx.costdists)
        t = @elapsed expand!(G, eval_i, ids, dists)
        v2 = distance_evaluations(ctx, before) / length(eval_i)      # expand! does not count: measured below by hand
        pq2 = perqueryscores(recallscore, gold_i, ids)
        nexp = isempty(G.members) ? 0.0 : mean(sum(length(members(G, r)) for r in unique(c -> c, ids[:, j]) if r != 0; init=0) for j in 1:length(eval_i))
        @printf("    bsize=%d Δ=%.1f: stage 1 recall %.4f (visits %.0f) | expanded recall %.4f (+%.0f member evaluations/query, %.1f ms total) | tied-gold queries: %.3f -> %.3f, failed %d -> %d\n",
                bsize, Δ, mean(pq1), v1, mean(pq2), nexp, 1000t, mean(pq1[tied]), mean(pq2[tied]), count(==(0), pq1[tied]), count(==(0), pq2[tied]))
    end
end

for (name, nd) in (("default (every object a node)", typemin(Float32)), ("members, neardup=0", 0f0))
    Random.seed!(1)
    G = SearchGraph(Dist.SqL2(), db)
    ctx = SearchGraphContext(; reporters=[], neighborhood=Neighborhood(; neardup=nd))
    t = @elapsed index!(G, ctx)
    @printf("%s: built in %.1f s, construction left %s\n", name, t, G.algo[])
    report(G, ctx, name); flush(stdout)
    # the tuning path with members: external queries, the default goal
    Random.seed!(2)
    tq = SubDatabase(MatrixDatabase(Qi[:, 1:500]), randperm(500)[1:256])
    t = @elapsed optimize_index!(G, ctx, MinRecall(0.9); queries=tq)
    ids, dists = searchbatch(G, ctx, eval_i, K); expand!(G, eval_i, ids, dists)
    @printf("    tuned in %.1f s to %s: expanded recall %.4f\n", t, G.algo[], macrorecall(gold_i, ids)); flush(stdout)
end
