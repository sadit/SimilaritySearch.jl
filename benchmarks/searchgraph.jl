using SimilaritySearch, SimilaritySearch.AdjacencyLists, Statistics, StatsBase, Random, JSON

function run(D, dist, db, queries, gold; logbase, ksearch, minrecall, minrecall_search, dim, exdim)
    algo = "SearchGraph"
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(
        neighborhood=Neighborhood(; filter=SatNeighborhood(), logbase),
        hyperparameters_callback=OptimizeParameters(MinRecall(minrecall)),
        parallel_block=2^13
    )
    buildtime = @elapsed index!(graph, ctx)
    opttime = @elapsed optimize_index!(graph, ctx, MinRecall(minrecall_search))
    searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch; sorted=false)
    recall = macrorecall(gold, knns)
    @info graph
    mem = sum(map(length, graph.adj.end_point)) * sizeof(eltype(graph.adj.end_point[1])) / 2^20 # adj mem
    n, m = length(db), length(queries)
    @show length(db) length(queries) mem recall searchtime
    @info "== QpS: $(m/searchtime)"
    N = [neighbors_length(graph.adj, i) for i in eachindex(graph.adj)]
    NQ = quantile(N, 0:0.1:1)
    @info "Neighborhood size quantiles: $NQ"
    push!(D, (; buildtime, opttime, searchtime, recall, mem, n, m, ksearch, minrecall, minrecall_search, logbase, quantile=NQ, dim, exdim))
end

function main_l2(D, n, m, dim, exdim;
    logbase=1.3f0, ksearch=16, minrecall=0.99, minrecall_search=0.9
)
    @info "=== n=$n m=$m dim=$dim exdim=$exdim ksearch=$ksearch"
    @assert dim <= exdim
    rng = Xoshiro(n)
    dist = SqL2Distance()
    #dist = SqL2_asf32()
    P = randn(rng, Float32, exdim, dim)

    db = let X = rand(rng, Float32, dim, n)
        MatrixDatabase(P * X)
    end

    queries = let X = rand(rng, Float32, dim, m)
        MatrixDatabase(P * X)
    end

    seq = ExhaustiveSearch(; dist, db)
    gold = searchbatch(seq, GenericContext(), queries, ksearch)
    run(D, dist, db, queries, gold; logbase, ksearch, minrecall, minrecall_search, dim, exdim)
end

D = []
#for n in [300, 10_000, 100_000], m in [1000], dim in [2, 4, 8, 16], exdim in [16, 32, 64, 128, 256, 512]
for n in [300, 100_000], m in [1000], dim in [2, 4, 8, 16], exdim in [256]
    m = min(m, n)
    main_l2(D, n, m, dim, exdim)
    @info D[end]
end

for r in D
    println(JSON.json(r))
end

