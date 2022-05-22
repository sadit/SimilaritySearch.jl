
using SimilaritySearch, Random
using Test

#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

@testset "vector indexing with SearchGraph" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    Random.seed!(0)
    ksearch = 10
    n, m, dim = 100_000, 100, 8

    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))

    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, db)
    goldI, goldD, goldtime = timedsearchbatch(seq, queries, ksearch)

    for bsize in [2, 12]
        search_algo = BeamSearch(; bsize)
        @info "=================== $search_algo"
        graph = SearchGraph(; db=DynamicMatrixDatabase(Float32, dim), dist, search_algo=search_algo)
        neighborhood = Neighborhood(reduce=IdentityNeighborhood())
        append!(graph, db; neighborhood, parallel_block=8, callbacks=SearchGraphCallbacks(hyperparameters=nothing))
        I, D, searchtime = timedsearchbatch(graph, queries, ksearch)
        #@info sort!(length.(graph.links), rev=true)
        @show goldD[:, 1]
        @show D[:, 1]
        @show goldI[:, 1]
        @show I[:, 1]
        recall = macrorecall(goldI, I)
        @info "testing search_algo: $(string(graph.search_algo)), time: $(searchtime)"
        @test recall >= 0.6
        @info "queries per second: $(1/searchtime), recall: $recall"
        @info "===="
    end

    @info "--- Optimizing parameters ParetoRadius ---"
    graph = SearchGraph(; dist, search_algo=BeamSearch(bsize=2), verbose=false)
    neighborhood = Neighborhood(reduce=DistalSatNeighborhood())
    append!(graph, db; neighborhood, callbacks=SearchGraphCallbacks(ParetoRadius()))
    @info "---- starting ParetoRadius optimization ---"
    optimize!(graph, ParetoRadius())
    I, D, searchtime = timedsearchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "ParetoRadius:> queries per second: ", 1/searchtime, ", recall:", recall
    @info graph.search_algo
    @test recall >= 0.3  # we don't expect high quality results on ParetoRadius


    @info "---- starting ParetoRecall optimization ---"
    optimize!(graph, ParetoRecall())
    I, D, searchtime = timedsearchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "ParetoRecall:> queries per second: ", 1/searchtime, ", recall:", recall
    @info graph.search_algo
    @test recall >= 0.6

    @info "========================= Callback optimization ======================"
    graph = SearchGraph(; db, dist, verbose=false)
    neighborhood = Neighborhood(reduce=DistalSatNeighborhood())
    index!(graph; neighborhood, callbacks=SearchGraphCallbacks(MinRecall(0.9)))
    optimize!(graph, MinRecall(0.9))
    I, D, searchtime = timedsearchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "testing without additional optimizations: queries per second:", 1/searchtime, ", recall: ", recall
    @info graph.search_algo
    @test recall >= 0.6

    @info "#############=========== Default parameters ==========###########"
    dim = 4
    db = MatrixDatabase(randn(Float32, dim, n))
    queries = MatrixDatabase(randn(Float32, dim, m))
    seq = ExhaustiveSearch(; dist, db)
    goldI, goldD = searchbatch(seq, queries, ksearch)
    graph = SearchGraph(; db, dist, verbose=false)
    index!(graph)
    searchtime = @elapsed I, D = searchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "testing without additional optimizations> queries per second:", m/searchtime, ", recall: ", recall
    @info graph.search_algo
    @test recall >= 0.7
end
