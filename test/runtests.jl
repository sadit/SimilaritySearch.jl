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

function test_vector_index(search_algo, neighborhood_algo, dist, ksearch)
    @testset "indexing with different algorithms" begin
        index = LocalSearchIndex(Vector{Float32}, dist, search=Nullable{LocalSearchAlgorithm}(search_algo), neighborhood=Nullable{NeighborhoodAlgorithm}(neighborhood_algo))
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
        res = search(index, rand(Float32, dim), KnnResult(ksearch))
        @show res
    end

    return index, length(res)
end

@testset "some vector indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    for search_algo in [IHCSearch(), NeighborhoodSearch(), BeamSearch()]
        for neighborhood_algo in [EssencialNeighborhood(), FixedNeighborhood(8), GallopingNeighborhood(), GallopingSatNeighborhood(), LogNeighborhood(), LogSatNeighborhood(), SatNeighborhood(), VorNeighborhood()]
            for dist in Any[L2SquaredDistance(), L2Distance(), L1Distance(), LInfDistance(), LpDistance(0.5)]
                index, numres = test_vector_index(search_algo, neighborhood_algo, dist, ksearch)
                acc += numres
                expected_acc += ksearch
            end
        end
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.95

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

function test_seq_index(search_algo, neighborhood_algo, dist, ksearch)
    @testset "indexing with different algorithms" begin
        index = LocalSearchIndex(Vector{Int}, dist, search=Nullable{LocalSearchAlgorithm}(search_algo), neighborhood=Nullable{NeighborhoodAlgorithm}(neighborhood_algo))
        index.options.verbose = false

        n = 100
        dim = 3
        function create_item()
            if search_algo isa JaccardDistance || search_algo isa DiceDistance || search_algo isa IntersectionDistance
                s = unique(rand(1:10, dim))
                sort!(s)
                return s
            else
                return rand(1:10, dim)
            end
        end
        info("inserting items to the index")
        for i in 1:n
            s = create_item()
            
            push!(index, s)
        end
        
        info("done; now testing")
        @test length(index.db) == n
        item = create_item()
        res = search(index, item, KnnResult(ksearch))
        @show res
    end

    return index, length(res)
end

@testset "some sequence distances indexing" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are requiered
    ksearch = 10
    acc = 0
    expected_acc = 0
    for search_algo in [BeamSearch()]
        for neighborhood_algo in [LogNeighborhood(1.5)]
            for dist in Any[JaccardDistance(), DiceDistance(), IntersectionDistance(), CommonPrefixDistance(), LevDistance(), LcsDistance(), HammingDistance()]
                index, numres = test_seq_index(search_algo, neighborhood_algo, dist, ksearch)
                acc += numres
                expected_acc += ksearch
            end
        end
    end

    # this is not really an error, but we test it anyway, it is more about the quality of the results
    @test acc / expected_acc > 0.95

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

    for BOW in [RBOW, HBOW]
        u1 = BOW(u)
        v1 = BOW(v)
        w1 = BOW(w)
        dist = AngleDistance()
        @test (dist(u1, v1), dist(u1, u1), dist(w1, u1)) == (0.5975474841801046, 0.0, 1.5707963267948966)
    end
end
