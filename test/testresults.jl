# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

@testset "Result set" begin
    k = 4
    V = rand(Float32, k*k)
    Vsorted = sort!([Item(i, v) for (i, v) in enumerate(V)])
    res = KnnResult(k)

    for i=1:length(V)
        push!(res, i, V[i])
    end

    @show Vsorted
    @test nearestdist(res) == Vsorted[1].dist
    arr = sortresults!(res)
    @test [x.id for x in arr] == [x.id for x in Vsorted[1:k]]
    @test [x.dist for x in arr] == [x.dist for x in Vsorted[1:k]]
    
end
