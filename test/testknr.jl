# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using LinearAlgebra
using Test

## 
## function test_knr_vectors(create_index, dist::PreMetric, ksearch)
##     index = create_index(db)
##     optimize!(index, dist, recall=0.9, k=10)
##     perf = Performance(dist, index.db, queries, expected_k=ksearch)
##     p = probe(perf, index, dist)
##     @show dist, p
##     @test p.recall > 0.8
## 
##     @info "adding more items"
##     for item in queries
##         push!(index, dist, item)
##     end
##     perf = Performance(dist, index.db, queries, expected_k=1)
##     p = probe(perf, index, dist)
##     @show dist, p
##     @test p.recall > 0.999
##     return p
## end

function test_knr_vectors(knr, queries, ksearch, gold)
    results = [search(knr, q, KnnResult(ksearch)) for q in queries]

    S = scores.(gold, results)
    s = scores(S)
    @show s
    @test s.macro_recall >= 0.9
end

@testset "indexing vectors with Knr" begin
    # NOTE: The following algorithms are complex enough to say we are testing
    # it doesn't have syntax errors and achieves some expected quality
    # a more grained test functions are required
    numrefs = 127
    kbuild = 3
    k = 10
    n = 10000 # number of items in the dataset
    m = 100  # number of queries
    dim = 3  # vector's dimension
    db = [rand(Float32, dim) |> normalize! for i in 1:n]
    queries = [rand(Float32, dim) |> normalize! for i in 1:m]

    dist = L2Distance()
    seq = ExhaustiveSearch(dist, db, k)
    gold = [search(seq, q, KnnResult(k)) for q in queries]

    knr = Knr(SqL2Distance(), db; numrefs=numrefs, kbuild=kbuild)
    test_knr_vectors(knr, queries, k, gold)
end

