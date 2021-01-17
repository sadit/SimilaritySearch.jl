# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test
#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function test_graph_vectors(graph, queries, ksearch, gold)
    results = [search(graph, q, KnnResult(ksearch)) for q in queries]

    S = scores.(gold, results)
    s = scores(S)
    @show s
    @test s.macro_recall >= 0.9
end

function create_graph(dist::PreMetric, search_algo, neighborhood_algo)
    index = fit(SearchGraph, dist, Vector{Float32}[], recall=0.9, search_algo=search_algo, neighborhood_algo=neighborhood_algo)
    n = 10_000
    dim = 3

    @info "inserting items to the index"
    for i in 1:n
        vec = rand(Float32, dim)
        push!(index, dist, vec)
    end
    
    @info "done; now testing"
    @test length(index.db) == n
    res = search(index, dist, rand(Float32, dim), KnnResult(ksearch))
    @show res
    return index, length(res)
end

@testset "some vector indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0

    for search_algo in [IHCSearch()] #[IHCSearch(), BeamSearch()]
        #for neighborhood_algo in [FixedNeighborhood(8), GallopingNeighborhood(), GallopingSatNeighborhood(), LogNeighborhood(), LogSatNeighborhood(), SatNeighborhood(), VorNeighborhood()]
        for neighborhood_algo in [FixedNeighborhood(8)]
            # for dist in Any[L2Distance(), L1Distance(), LInftyDistance(), LpDistance(0.5)]
            dist = SqL2Distance()
            @testset "indexing vectors with SearchGraph and $dist" begin
                index, numres = test_index(dist, search_algo, neighborhood_algo, ksearch)
                acc += numres
                expected_acc += ksearch
            end
        end
    end
end
