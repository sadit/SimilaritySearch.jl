# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using LinearAlgebra
using Test

function test_knr_vectors(perf, knr)
    @test probe(perf, knr).macrorecall >= 0.8
end

@testset "indexing vectors with Knr" begin
    # NOTE: The following algorithms are complex enough to say we are testing
    # it doesn't have syntax errors and achieves some expected quality
    # a more grained test functions are required
    k = 10
    n = 10000 # number of items in the dataset
    m = 100  # number of queries
    dim = 3  # vector's dimension
    db = [rand(Float32, dim) |> normalize! for i in 1:n]
    queries = [rand(Float32, dim) |> normalize! for i in 1:m]

    dist = L2Distance()
    seq = ExhaustiveSearch(dist, db, k)
    perf = Performance(seq, queries, k)
    for numrefs in [64], kbuild in [3]
        knr = Knr(dist, db; numrefs=numrefs, kbuild=kbuild)
        test_knr_vectors(perf, knr)
    end
end

