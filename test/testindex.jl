#
# This file contains a set of tests for LocalSearchIndex over databases of #sequences
#

using SimilaritySearch

function test_index(dist, ksearch)
    @testset "indexing with different algorithms" begin
        n = 1000
        dim = 3
        info("inserting items to the index")
        db = Vector{Int}[]
    
        function create_item()
            s = unique(rand(1:10, dim))
            if dist isa JaccardDistance || dist isa DiceDistance || dist isa IntersectionDistance
                sort!(s)
            end
            return s
        end
        info("inserting items to the index")
        for i in 1:n
            s = create_item()            
            push!(db, s)
        end
        index = Knr(db, dist, 100, 7)
        
        info("done; now testing")
        @test length(index.db) == n
        item = create_item()
        res = search(index, item, KnnResult(ksearch))
        @show res
        @show index.invindex[1]
        return index, length(res)
    end
end

@testset "some sequence distances indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    local index

    for dist in Any[JaccardDistance(), DiceDistance(), IntersectionDistance(), CommonPrefixDistance(), LevDistance(), LcsDistance(), HammingDistance()]
        index, numres = test_index(dist, ksearch)
        acc += numres
        expected_acc += ksearch
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.8
end
