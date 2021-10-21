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
    n, m, dim = 1000, 30, 3

    db = [rand(Float32, dim) for i in 1:n]
    queries = [rand(Float32, dim) for i in 1:m]

    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, db)
    perf = Performance(seq, queries, ksearch)
    
    for search_algo_fun in [() -> IHCSearch(restarts=4), () -> BeamSearch(bsize=4)]
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
end
