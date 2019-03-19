using SimilaritySearch
using SimilaritySearch.Graph
using Test
#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

function test_index_search_with_hint(dist::Function, ksearch::Int, search_algo, neighborhood_algo)
    @testset "indexing with different algorithms" begin
        index = fit(SearchGraph, dist, Vector{Float32}[], search_algo=search_algo, neighborhood_algo=neighborhood_algo)

        n = 10_000
        dim = 3
        @info "inserting items to the index"
        for i in 1:n
            vec = rand(Float32, dim)
            push!(index, dist, vec)
        end
        
        @info "done; now testing with hint"

        m = 1
        
        start = time()
        for i in 1:m
            q = index.db[i]
            #search(index, dist, q, KnnResult(ksearch), hints=Int.(index.links[i]))
            search(index, dist, q, KnnResult(ksearch), hints=Int.(index.links[i]))
        end

        @info "Hints=true, noise=false: Query time $((time() - start) / m)"

        start = time()
        for i in 1:m
            q = index.db[i] + rand(Float32, dim) .* Float32(0.0001)
            search(index, dist, q, KnnResult(ksearch), hints=Int.(index.links[i]))
        end

        @info "Hints=true, noise=true: Query time $((time() - start) / m)"

        start = time()
        for i in 1:m
            q = index.db[i]
            search(index, dist, q, KnnResult(ksearch))
        end
        @info "Hints=false, noise=false: Query time $((time() - start) / m)"

        start = time()
        for i in 1:m
            q = index.db[i] + rand(Float32, dim) .* Float32(0.0001)
            search(index, dist, q, KnnResult(ksearch))
        end
        @info "Hints=false, noise=true: Query time $((time() - start) / m)"

    end
end

function test_index(dist::Function, ksearch::Int, search_algo, neighborhood_algo)
    @testset "indexing with different algorithms" begin
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
end

@testset "some vector indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0

    for search_algo in [TIHCSearch(), IHCSearch(), BeamSearch()]
        for neighborhood_algo in [SatNeighborhood()]
        #for neighborhood_algo in [EssencialNeighborhood(), FixedNeighborhood(8), GallopingNeighborhood(), GallopingSatNeighborhood(), LogNeighborhood(), LogSatNeighborhood(), SatNeighborhood(), VorNeighborhood()]
            # for dist in Any[l2_distance, L2Distance(), L1Distance(), LInfDistance(), LpDistance(0.5)]
            dist = l2_distance
            index, numres = test_index(dist, ksearch, search_algo, neighborhood_algo)
            acc += numres
            expected_acc += ksearch
        end
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    # @test acc / expected_acc > 0.9

    test_index_search_with_hint(l2_distance, ksearch, BeamSearch(), SatNeighborhood())
    # @show "Showing AKNN ($k)"
    ## n = length(index.db)
    ## k = 3
    ## aknn = compute_aknn(index, l2_distance, k)
    ## @test n == length(aknn)
    ## for p in aknn
    ##     @test length(p) > 0
    ## end
end
