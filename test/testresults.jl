# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

@testset "Result set" begin
    k=4
    V = rand(Float32, k*k)
    Vsorted = sort(V)[1:k]

    res = KnnResult(k)
    sres = KnnResult(k)
    nn = KnnResult(1)

    for i=1:length(V)
        push!(res, i, V[i])
        push!(sres, i, V[i])
        push!(nn, i, V[i])
    end
    @test [x.objID for x in res] == [x.objID for x in sres]
    @test [x.dist for x in res] == [x.dist for x in sres] == Vsorted
    @test first(nn).dist == Vsorted[1]
end
