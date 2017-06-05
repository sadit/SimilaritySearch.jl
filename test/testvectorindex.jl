
function test_vector_index(search_algo, neighborhood_algo, dist, ksearch)
    @testset "indexing with different algorithms" begin
        index = LocalSearchIndex(Vector{Float32}, dist, search=search_algo, neighborhood=neighborhood_algo)
        index.options.verbose = false

        n = 100
        dim = 3
        info("inserting items to the index")
        for i in 1:n
            vec = rand(Float32, dim)
            # NNS.fit!(index, V)
            push!(index, vec)
        end
        
        info("done; now testing")
        @test length(index.db) == n
        res = search(index, rand(Float32, dim), KnnResult(ksearch))
        @show res
    end

    return index, length(res)
end


@testset "some vector indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    for search_algo in [IHCSearch(), NeighborhoodSearch(), BeamSearch()]
        for neighborhood_algo in [EssencialNeighborhood(), FixedNeighborhood(8), GallopingNeighborhood(), GallopingSatNeighborhood(), LogNeighborhood(), LogSatNeighborhood(), SatNeighborhood(), VorNeighborhood()]
            for dist in Any[L2SquaredDistance(), L2Distance(), L1Distance(), LInfDistance(), LpDistance(0.5)]
                index, numres = test_vector_index(search_algo, neighborhood_algo, dist, ksearch)
                acc += numres
                expected_acc += ksearch
            end
        end
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.9

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
