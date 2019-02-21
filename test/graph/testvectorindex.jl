using SimilaritySearch
using SimilaritySearch.NearNeighborGraph
using Test

#
# This file contains a set of tests for LocalSearchIndex over databases of vectors (of Float32)
#


function test_index_search_at(search_algo, neighborhood_algo, dist, ksearch)
    @testset "indexing with different algorithms" begin
        index = LocalSearchIndex(Vector{Float32}, dist, search=search_algo, neighborhood=neighborhood_algo)

        n = 100
        dim = 2
        @info "inserting items to the index"
        for i in 1:n
            vec = rand(Float32, dim)
            push!(index, vec)
        end
        
        @info "done; now testing"
        @test length(index.db) == n
        res = search_at(index, index.db[1] + rand(Float32, dim) .* Float32(0.001), 1, KnnResult(ksearch))
        res = search_at(index, index.db[2] + rand(Float32, dim) .* Float32(0.001), 2, KnnResult(ksearch))
        @show res
        return index, length(res)
    end
end

function test_index(search_algo, neighborhood_algo, dist, ksearch)
    @testset "indexing with different algorithms" begin
        index = LocalSearchIndex(Vector{Float32}, dist, search=search_algo, neighborhood=neighborhood_algo)

        n = 100
        dim = 3
        @info "inserting items to the index"
        for i in 1:n
            vec = rand(Float32, dim)
            # NNS.fit!(index, V)
            push!(index, vec)
        end
        
        @info "done; now testing"
        @test length(index.db) == n
        res = search(index, rand(Float32, dim), KnnResult(ksearch))
        @show res
        return index, length(res)
    end
end


@testset "some vector indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    local index 
    for search_algo in [IHCSearch(), NeighborhoodSearch(), BeamSearch(), DeltaSearch(), ShrinkingNeighborhoodSearch()]
        for neighborhood_algo in [EssencialNeighborhood(), FixedNeighborhood(8), GallopingNeighborhood(), GallopingSatNeighborhood(), LogNeighborhood(), LogSatNeighborhood(), SatNeighborhood(), VorNeighborhood()]
            # for dist in Any[L2SquaredDistance(), L2Distance(), L1Distance(), LInfDistance(), LpDistance(0.5)]
            dist = L2SquaredDistance()
            index, numres = test_index(search_algo, neighborhood_algo, dist, ksearch)
            acc += numres
            expected_acc += ksearch
        end
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.9

    index, numres = test_index_search_at(BeamSearch(), FixedNeighborhood(), L2SquaredDistance(), ksearch)
    n = length(index.db)
    k = 3
    @show "Showing AKNN ($k)"
    aknn = compute_aknn(index, k)
    @test n == length(aknn)
    for p in aknn
        @show p
        @test length(p) > 0
    end
end
