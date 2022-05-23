
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
    goldtime = @elapsed goldI, goldD = searchbatch(seq, queries, ksearch)

    for bsize in [2, 12]
        search_algo = BeamSearch(; bsize)
        @info "=================== $search_algo"
        graph = SearchGraph(; db=DynamicMatrixDatabase(Float32, dim), dist, search_algo=search_algo)
        neighborhood = Neighborhood(reduce=IdentityNeighborhood())
        append!(graph, db; neighborhood, parallel_block=8, callbacks=SearchGraphCallbacks(hyperparameters=nothing))
        searchtime = @elapsed I, D= searchbatch(graph, queries, ksearch)
        #@info sort!(length.(graph.links), rev=true)
        @show goldD[:, 1]
        @show D[:, 1]
        @show goldI[:, 1]
        @show I[:, 1]
        recall = macrorecall(goldI, I)
        @info "testing search_algo: $(string(graph.search_algo)), time: $(searchtime)"
        @test recall >= 0.6
        @info "queries per second: $(m/searchtime), recall: $recall"
        @info "===="
    end

    @info "--- Optimizing parameters ParetoRadius ---"
    graph = SearchGraph(; dist, search_algo=BeamSearch(bsize=2), verbose=false)
    neighborhood = Neighborhood(reduce=DistalSatNeighborhood())
    append!(graph, db; neighborhood, callbacks=SearchGraphCallbacks(ParetoRadius()))
    @info "---- starting ParetoRadius optimization ---"
    optimize!(graph, ParetoRadius())
    searchtime = @elapsed I, D = searchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "ParetoRadius:> queries per second: ", m/searchtime, ", recall:", recall
    @info graph.search_algo
    @test recall >= 0.3  # we don't expect high quality results on ParetoRadius


    @info "---- starting ParetoRecall optimization ---"
    optimize!(graph, ParetoRecall())
    searchtime = @elapsed I, D = searchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "ParetoRecall:> queries per second: ", m/searchtime, ", recall:", recall
    @info graph.search_algo
    @test recall >= 0.6

    @info "========================= REBUILD TEST ======================"
    graph = SearchGraph(; db, dist, verbose=false)
    index!(graph; callbacks=SearchGraphCallbacks(MinRecall(0.9)))
    optimize!(graph, MinRecall(0.9); queries)
    searchtime = @elapsed I, D = searchbatch(graph, queries, ksearch)
    recall = macrorecall(goldI, I)
    @info "testing without additional optimizations: queries per second:", m/searchtime, ", recall: ", recall
    @info graph.search_algo
    @test recall >= 0.6

    @info "========================= rebuild process =========================="
    graph = rebuild(graph)
    optimize!(graph, MinRecall(0.9); queries)  # using the actual dataset makes prone to overfitting hyperparameters (more noticeable in rebuilt indexes)
    searchtime_ = @elapsed I, D = searchbatch(graph, queries, ksearch)
    recall_ = macrorecall(goldI, I)

    @info "-- old vs rebuild> searchtime: $searchtime vs $searchtime_; recall: $recall vs $recall_"

    @info "#############=========== Default parameters (useful as fast benchmark) ==========###########"
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
