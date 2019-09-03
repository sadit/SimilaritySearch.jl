using SimilaritySearch
using Test
#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

dist = l2_distance

function create_dataset()
    n = 10000
    dim = 2
    [rand(Float32, dim) for i in 1:n]
end

@testset "All K-NN Sequential" begin
    db = create_dataset()
    @time graph = fit(SearchGraph, dist, db)
    index = fit(Sequential, db)
    k = 3
    @time A = allknn(index, dist, db, k=k)
    @time B = allknn(graph, dist, db, k=k)

    n = length(A)
    f = 1 / n
    s = 0.0
    for i in 1:n
        a = Set(p.objID for p in A[i])
        b = Set(p.objID for p in B[i])
        # @show a b intersect(a, b)
        s += f * length(intersect(a, b)) / k
    end

    @show f s
end

