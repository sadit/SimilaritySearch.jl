using NNS
using Base.Test

# write your own tests here

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

function test_index(search_algo)
    index = LocalSearchIndex()

    @testset "indexing with different search algorithms" begin
        index.search_algo = search_algo()
        index.options.verbose = false
        n = 100
        dim = 3
        info("inserting items to the index")
        for i in 1:n
            vec = rand(Float32, dim)
            # NNS.fit!(index, V)
            push!(index, vec)
        end
        
        info("done; now testing")
        @test length(index.db) == n
        res = search(index, rand(Float32, dim), KnnResult(10))
        @show res
        @test length(res) == 10    
    end

    return index
end

@testset "some indexing" begin
    for search_algo in [IHCSearch, NeighborhoodSearch, BeamSearch]
        index = test_index(search_algo)
    end

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


@testset "DocumentType and RBOW" begin
    u = Dict("el" => 0.9, "hola" => 0.1, "mundo" => 0.2)
    v = Dict("el" => 0.4, "hola" => 0.2, "mundo" => 0.4)
    w = Dict("xel" => 0.4, "xhola" => 0.2, "xmundo" => 0.4)
    u1 = RBOW(u)
    v1 = RBOW(v)
    w1 = RBOW(w)
    dist = AngleDistance()
    @test (dist(u1, v1), dist(u1, u1), dist(w1, u1)) == (0.5975474841801046, 0.0, 1.5707963267948966)
end

