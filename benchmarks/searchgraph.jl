using SimilaritySearch, SimilaritySearch.AdjacencyLists, Statistics, StatsBase

function run(dist, db, queries, gold, ksearch; nndist=0.01f0, logbase=1.3f0)
    graph = SearchGraph(; db, dist)
    ctx = SearchGraphContext(
        neighborhood=Neighborhood(; filter=SatNeighborhood(; nndist), logbase),
        hyperparameters_callback=OptimizeParameters(MinRecall(0.99)),
        parallel_block=256
    )
    @time "INDEXING" index!(graph, ctx)
    @time "INDEX OPT" optimize_index!(graph, ctx, MinRecall(0.9))
    searchtime = @elapsed knns = searchbatch(graph, ctx, queries, ksearch; sorted=false)
    recall = macrorecall(gold, knns)
    G = matrixhints(graph, MatrixDatabase)
    @info graph
    @info G
    searchtimeG = @elapsed knns = searchbatch(G, ctx, queries, ksearch; sorted=false)
    recallG = macrorecall(gold, knns)
    mem = sum(map(length, graph.adj.end_point)) * sizeof(eltype(graph.adj.end_point[1])) / 2^20 # adj mem
    m = length(queries)
    @show length(db) length(queries) mem recall recallG searchtime searchtimeG
    @info "== QpS (orig): $(m/searchtime), QpS (G): $(m/searchtimeG)"
    N = [neighbors_length(graph.adj, i) for i in eachindex(graph.adj)]
    NQ = quantile(N, 0:0.1:1)
    @info "Neighborhood size quantiles: $NQ"
    graph
end

function main(n, m;
    dim=8,
    ksearch=32,
    dist=SqL2Distance()
)
    db = MatrixDatabase(randn(Float32, dim, n))
    queries = MatrixDatabase(randn(Float32, dim, m))
    seq = ExhaustiveSearch(; dist, db)
    gold = searchbatch(seq, GenericContext(), queries, ksearch)
    run(dist, db, queries, gold, ksearch)
end

@info "===================== WARMING ======================"
main(300, 100)
@info "===================== REAL TEST =========================="
main(10^6, 10000)
