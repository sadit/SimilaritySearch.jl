# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test
using JSON
#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    n, m, dim = 10_000, 100, 4

    db = [rand(Float32, dim) for i in 1:n]
    queries = [rand(Float32, dim) for i in 1:m]

    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, db)
    perf = Performance(seq, queries, ksearch)

    
    for neighborhood_algo_fun in [
                () -> FixedNeighborhood(8), ## objects need to be created at each usage here since 
                () -> LogNeighborhood(), ## since they contain state data
                () -> LogSatNeighborhood(),
                () -> SatNeighborhood(),
                () -> VorNeighborhood()
        ], search_algo_fun in [
            () -> IHCSearch(4)  
            () -> BeamSearch()  
            ]
        @info "==================="
        graph = SearchGraph(dist, db;
            search_algo=search_algo_fun(),
            neighborhood_algo=neighborhood_algo_fun(),
            automatic_optimization=false)
        @time p = probe(perf, graph)
        @info "testing search_algo: $(string(graph.search_algo)), neighborhood_algo: $(graph.neighborhood_algo), p: $(JSON.json(p))"
        @test p.macrorecall >= 0.6
        @info "queries per second: $(1/p.searchtime)"
        @info "===="
        graph = SearchGraph(dist, db;
            search_algo=search_algo_fun(),
            neighborhood_algo=neighborhood_algo_fun(),
            automatic_optimization=true, 
            recall=0.9,
            verbose=false)
        @time p = probe(perf, graph)
        @info "testing search_algo: $(string(graph.search_algo)), neighborhood_algo: $(graph.neighborhood_algo), p: $(JSON.json(p)); using automatic optimization"
        @test p.macrorecall >= 0.7
        @info "queries per second: $(1/p.searchtime)"
    end
end
