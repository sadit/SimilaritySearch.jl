# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Random
using Test

#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    Random.seed!(0)
    ksearch = 10
    n, m, dim = 100_000, 100, 4

    db = [rand(Float32, dim) for i in 1:n]
    queries = [rand(Float32, dim) for i in 1:m]

    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, db)
    perf = Performance(seq, queries, ksearch)
    
    for search_algo_fun in [() -> IHCSearch(restarts=4), () -> BeamSearch(bsize=2)]
        search_algo = search_algo_fun()
        @info "=================== $search_algo"
        graph = SearchGraph(; dist, search_algo=search_algo)
        graph.neighborhood.reduce = SatNeighborhood()
        append!(graph, db)
        @time p = probe(perf, graph)
        @info "testing search_algo: $(string(graph.search_algo)), p: $(p)"
        @test p.macrorecall >= 0.6
        @info "queries per second: $(1/p.searchtime)"
        @info "===="
    end

    @info "--- Optimizing parameters :pareto_distance_searchtime ---"
    graph = SearchGraph(; dist, search_algo=BeamSearch(bsize=2), verbose=false)
    graph.neighborhood.reduce = SatNeighborhood()
    append!(graph, db)
    @info "---- starting :pareto_distance_searchtime optimization ---"
    optimize!(graph, OptimizeParameters())
    @time p = probe(perf, graph)
    @info ":pareto_distance_search_time: $p ; queries per second:", 1/p.searchtime
    @info graph.search_algo
    @test p.macrorecall >= 0.6

    @info "---- starting :pareto_recall_searchtime optimization ---"
    optimize!(graph, OptimizeParameters(kind=:pareto_recall_searchtime))
    @time p = probe(perf, graph)
    @info ":pareto_recall_search_time: $p ; queries per second:", 1/p.searchtime
    @info graph.search_algo
    @test p.macrorecall >= 0.6

    @info "========================= Callback optimization ======================"
    @info "--- Optimizing parameters :pareto_distance_searchtime ---"
    graph = SearchGraph(; dist, search_algo=BeamSearch(bsize=2), verbose=false)
    graph.neighborhood.reduce = SatNeighborhood()
    graph.callbacks[:optimization] = OptimizeParameters(kind=:pareto_distance_searchtime)
    append!(graph, db)
    @time p = probe(perf, graph)
    @info "testing without additional optimizations: $p ; queries per second:", 1/p.searchtime
    @info graph.search_algo
    @test p.macrorecall >= 0.6

    
    @info "#############=========== Callback optimization 2 ==========###########"
    @info "--- Optimizing parameters :pareto_distance_searchtime L2 with large norm ---"
    dim = 8
    db = [rand(1:100, dim) for i in 1:n]
    queries = [rand(1:100, dim) for i in 1:m]
    seq = ExhaustiveSearch(dist, db)
    perf = Performance(seq, queries, ksearch)

    graph = SearchGraph(; dist, search_algo=BeamSearch(bsize=2), verbose=false)
    graph.neighborhood.reduce = SatNeighborhood()
    graph.callbacks[:optimization] = OptimizeParameters(kind=:pareto_distance_searchtime)
    append!(graph, db)
    @time p = probe(perf, graph)
    @info "testing without additional optimizations: $p ; queries per second:", 1/p.searchtime
    @info graph.search_algo
    @test p.macrorecall >= 0.6
    
end
