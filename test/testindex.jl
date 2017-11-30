#
# This file contains a set of tests for LocalSearchIndex over databases of #sequences
#

using SimilaritySearch

function test_index(dist, ksearch)
    @testset "indexing with different algorithms" begin
        n = 1000
        dim = 7
        σ = 127
        κ = 3
        V = collect(1:30)
        function create_item()
            s = unique(rand(V, dim))
            if dist isa JaccardDistance || dist isa DiceDistance || dist isa IntersectionDistance
                sort!(s)
            end
            return s
        end
        info("inserting items into the index")
        db = Vector{Vector{Int}}(n)
        for i in 1:n
            db[i] = create_item()
        end
        index = Knr(db, dist, numrefs=σ, k=κ)
        optimize!(index, recall=0.9, use_distances=true)
        info("done; now testing")
        info("#refs:", index.refs |> length, ", #n:", length(index.db))
        @test length(index.db) == n
        item = create_item()
        res = search(index, item, KnnResult(ksearch))
        @show res
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
