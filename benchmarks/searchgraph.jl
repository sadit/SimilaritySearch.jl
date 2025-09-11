using JLD2, SimilaritySearch, SimilaritySearch.AdjacencyLists, Statistics, StatsBase

function run(dist, db, queries, gold, ksearch; nndist=0.01f0, logbase=1.3)
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(
        neighborhood = Neighborhood(; filter=SatNeighborhood(; nndist), logbase),
        #neighborhood = Neighborhood(filter=DistalSatNeighborhood(), logbase=1.3),
        #hyperparameters_callback = OptimizeParameters(OptRadius(0.01)),
        hyperparameters_callback = OptimizeParameters(MinRecall(0.95)),
        parallel_block = 256
    )
    @time "INDEXING" index!(graph, ctx)
    @time "INDEX OPT" optimize_index!(graph, ctx, MinRecall(0.9))
    searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
    searchtime2 = @elapsed knns = searchbatch(graph, ctx, queries, ksearch)
    recall = macrorecall(gold, knns)
    G = matrixhints(graph, StrideMatrixDatabase)
    searchtime3 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    searchtime4 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    recall_ = macrorecall(gold, knns)
    @assert recall == recall_
    searchtime5 = @elapsed knns = searchbatch(G, ctx, queries, ksearch)
    mem = sum(map(length, graph.adj.end_point)) * sizeof(eltype(graph.adj.end_point[1])) / 2^20 # adj mem
    m = length(queries)
    @show length(db) length(queries) mem recall
    @info "A> QpS: $(m/searchtime), QpS (already compiled): $(m/searchtime2)"
    @info "B> QpS: $(m/searchtime3), QpS (already compiled): $(m/searchtime5)"
    N = [neighbors_length(graph.adj, i) for i in eachindex(graph.adj)]
    Q = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.91, 0.92, 0.94, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 1.0]
    NQ = quantile(N, Q)
    @info "Neighborhood size quantiles:" [q => n for (q, n) in zip(Q, NQ)]
    @info graph.algo
    graph
end

function main(;
        dim = 4,
        n = 10^5,
        m = 100,
        ksearch = 10,
        dist = SqL2Distance()
    )
    db = StrideMatrixDatabase(randn(Float32, dim, n))
    queries = StrideMatrixDatabase(randn(Float32, dim, m))
    seq = ExhaustiveSearch(; dist, db)
    gold = searchbatch(seq, GenericContext(), queries, ksearch)
    run(dist, db, queries, gold, ksearch)
end

main()