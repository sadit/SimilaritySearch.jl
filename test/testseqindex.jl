function test_seq_index(search_algo, neighborhood_algo, dist, ksearch)
    @testset "indexing with different algorithms" begin
        index = LocalSearchIndex(Vector{Int}, dist, search=Nullable{LocalSearchAlgorithm}(search_algo), neighborhood=Nullable{NeighborhoodAlgorithm}(neighborhood_algo))
        index.options.verbose = false

        n = 100
        dim = 3
        function create_item()
            if search_algo isa JaccardDistance || search_algo isa DiceDistance || search_algo isa IntersectionDistance
                s = unique(rand(1:10, dim))
                sort!(s)
                return s
            else
                return rand(1:10, dim)
            end
        end
        info("inserting items to the index")
        for i in 1:n
            s = create_item()            
            push!(index, s)
        end
        
        info("done; now testing")
        @test length(index.db) == n
        item = create_item()
        res = search(index, item, KnnResult(ksearch))
        @show res
    end

    return index, length(res)
end

@testset "some sequence distances indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    for search_algo in [BeamSearch()]
        for neighborhood_algo in [LogNeighborhood(1.5)]
            for dist in Any[JaccardDistance(), DiceDistance(), IntersectionDistance(), CommonPrefixDistance(), LevDistance(), LcsDistance(), HammingDistance()]
                index, numres = test_seq_index(search_algo, neighborhood_algo, dist, ksearch)
                acc += numres
                expected_acc += ksearch
            end
        end
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.8

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
