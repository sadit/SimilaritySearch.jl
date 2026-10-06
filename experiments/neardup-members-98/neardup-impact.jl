# Exact duplicates in ccnews and what three ways of linking them do to the graph. Written before
# members existed: then Neighborhood(neardup=0f0) only thinned the clique with the HSP filter's
# decaying capture, and the tie rule was monkeypatched in. Today neardup=0f0 folds the duplicates
# into members (see members-impact.jl), so the three variants here are: the default (ties keep,
# cliques), members, and the tie rule alone (a duplicate as a degree-1 node, reachable through
# its twin's reverse link, no members table).
#
# usage: julia -t auto --project=<env with HDF5, SHA and this SimilaritySearch> neardup-impact.jl
using SimilaritySearch, HDF5, Statistics, Random, Dates, Printf, SHA
using SimilaritySearch: hsp_should_push, iterate_hsp_
const K = 10
path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}
Qi = h5read(path, "itest/queries"); Gi = h5read(path, "itest/knns")[1:K, :]
dim, n = size(X)

# ---- census of exact duplicates (byte-identical columns)
groups = Dict{Vector{UInt8},Vector{Int}}()
for i in 1:n
    push!(get!(groups, sha1(reinterpret(UInt8, view(X, :, i))), Int[]), i)
end
sizes = sort([length(g) for g in values(groups) if length(g) > 1]; rev=true)
@printf("exact-duplicate clusters: %d, covering %d points (%.2f%%); sizes >= 10: %d; largest: %s\n",
        length(sizes), sum(sizes), 100sum(sizes) / n, count(>=(10), sizes), join(sizes[1:min(8, end)], ","))
indup = falses(n); for g in values(groups); length(g) > 1 && (indup[g] .= true); end

db = MatrixDatabase(X)
eval_i = MatrixDatabase(Qi[:, 501:end]); gold_i = Gi[:, 501:end]
E = ExhaustiveSearch(Dist.SqL2(), db)
_, goldD = searchbatch(E, GenericContext(), eval_i, K)
tied = findall(j -> goldD[K, j] - goldD[1, j] == 0, 1:size(goldD, 2))          # fully tied gold neighborhoods
zero = findall(j -> goldD[K, j] == 0, 1:size(goldD, 2))                         # all gold at distance 0: the query has >= 10 exact copies
@printf("held-out queries: %d; fully tied gold: %d; all gold at distance 0: %d\n", length(eval_i), length(tied), length(zero))

# ---- the tie rule, patched in: a kept neighbor at distance 0 from the center rejects on <=
const TIERULE = Ref(false)
@eval SimilaritySearch function hsp_should_push(hsp_neighborhood::AbstractKnnQueue, dist::PreMetric, db::AbstractDatabase, center, point_id::UInt32, dist_center_point::Float32; factor::Float32=1.0f0, neardup::Float32=typemin(Float32))
    @inbounds point = db[point_id]
    @inbounds for p in IdDistView(hsp_neighborhood)
        dist_point_hsp = evaluate(dist, point, db[p.id])
        dist_point_hsp < dist_center_point && return false
        Main.TIERULE[] && p.dist == 0 && dist_point_hsp <= dist_center_point && return false
    end
    true
end

function evalgraph(G, ctx, name)
    adj = G.adj
    degs = [neighbors_length(adj, i) for i in 1:length(G)]
    @printf("  %s: edges %d, mean degree %.1f, max degree %d, degree-1 nodes %d (%.2f%%), mean degree of duplicate nodes %.1f\n",
            name, sum(degs), mean(degs), maximum(degs), count(==(1), degs), 100count(==(1), degs) / length(G), mean(degs[indup]))
    for (bsize, Δ) in ((8, 1.0), (16, 1.2))
        G.algo[] = BeamSearch(; bsize, Δ)
        before = copy(ctx.costdists)
        ids, dists = searchbatch(G, ctx, eval_i, K)
        visits = distance_evaluations(ctx, before) / length(eval_i)
        expand!(G, eval_i, ids, dists)                 # the raw neighbors behind the representatives (no-op without members)
        pq = perqueryscores(recallscore, gold_i, ids)
        @printf("    bsize=%d Δ=%.1f: recall %.4f, visits %.0f | tied-gold queries (%d): recall %.3f, failed (recall 0) %d | all-zero-gold queries (%d): recall %.3f, failed %d\n",
                bsize, Δ, mean(pq), visits, length(tied), mean(pq[tied]), count(==(0), pq[tied]), length(zero), isempty(zero) ? NaN : mean(pq[zero]), count(==(0), pq[zero]))
    end
end

for (name, tie, nd) in (("baseline (ties keep)", false, typemin(Float32)), ("members, neardup=0", false, 0f0), ("tie rule alone", true, typemin(Float32)))
    TIERULE[] = tie
    Random.seed!(1)
    G = SearchGraph(Dist.SqL2(), db)
    ctx = SearchGraphContext(; reporters=[], neighborhood=Neighborhood(; neardup=nd))
    t = @elapsed index!(G, ctx)
    @printf("%s: built in %.1f s, construction left %s\n", name, t, G.algo[])
    evalgraph(G, ctx, name)
    flush(stdout)
end
