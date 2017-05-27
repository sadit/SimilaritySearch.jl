@testset "Result set implementations" begin
    k=4
    V = rand(Float32, k*k)
    Vsorted = sort(V)[1:k]
    
    res = KnnResult(k)
    sres = SlugKnnResult(k)
    nn = NnResult()
    
    for i=1:length(V)
        # info("push!($i, $((V[i])))")
        push!(res, i, V[i])
        push!(sres, i, V[i])
        push!(nn, i, V[i])
    end
    
    @test [x.objID for x in res] == [x.objID for x in sres]
    @test [x.dist for x in res] == [x.dist for x in sres] == Vsorted
    @test first(nn).dist == Vsorted[1]
end
