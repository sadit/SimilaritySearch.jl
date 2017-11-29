#
# This file contains a set of tests for LocalSearchIndex over databases of vectors (of Float32)
# 

using SimilaritySearch

function test_vector_index(dist, ksearch)
    @testset "indexing with different algorithms" begin
        n = 1000
        dim = 2
        σ = 127
        κ = 3
        info("inserting items into the index")
        db = Vector{Vector{Float32}}(n)        
        for i in 1:n
            db[i] = rand(Float32, dim)
        end
    
        index = Knr(db, dist, σ, κ)
        optimize!(index, recall=0.9)
        info("done; now testing")
        @test length(index.db) == n
        res = search(index, rand(Float32, dim), KnnResult(ksearch))
        @show res
        @show index.invindex[1]
        return index, length(res)
    end
end

@testset "some vector indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    local index 

    #for dist in Any[L2SquaredDistance(), L2Distance(), L1Distance(), LInfDistance(), LpDistance(0.5)]
    for dist in Any[LInfDistance(), LpDistance(0.5)]
        index, numres = test_vector_index(dist, ksearch)
        acc += numres
        expected_acc += ksearch
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.9
end
