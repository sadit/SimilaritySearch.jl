using SimilaritySearch.Intersections
using Test, Random

Random.seed!(0)

@testset "Search" begin
    for i in 1:1000
        L = sort!(unique(rand(1:1000, i)))
        v = rand(L)
        p = findfirst(x -> x == v, L)

        for salgo in (binarysearch, doublingsearch, doublingsearchrev, seqsearch, seqsearchrev)
            pos = salgo(L, v)
            @test p == pos #&& error("gold:$p != found:$pos; L=$L, searching for: $v")
        end

        v = 0
        for salgo in (binarysearch, doublingsearch, doublingsearchrev, seqsearch, seqsearchrev)
            @test 1 == salgo(L, v)
        end

        v = 2000
        for salgo in (binarysearch, doublingsearch, doublingsearchrev, seqsearch, seqsearchrev)
            @test length(L)+1 == salgo(L, v)
        end

    end
end

randset(n, m) = [sort!(unique(rand(1:n, 100))) for _ in 1:m]

@testset "SVS" begin
    @test [1, 3, 5] == svs([[1, 2, 3, 4, 5], [1, 3, 5]])
    
    for ialgo2 in [baezayates!, imerge2!]
        n = 300
        for i in 1:n
            L = randset(n, 3)
            I = sort(intersect(L...))
            #@info (i, I, ialgo2)
            Lc = copy(L)
            S = svs(Lc, ialgo2)
            @test I == S
        end
    end
end

function doL(mergefun, L, salgo=doublingsearch)
    output = Int[]
    mergefun(output, copy(L), salgo)
    output
end

function doT(mergefun, L, t)
    output = Int[]
    mergefun(output, copy(L); t)
    output
end

@testset "BK" begin
    LIST = [[1, 2, 3, 4, 5, 6], [1, 3, 5]]
    @test [1, 3, 5] == doL(bk!, LIST)
    @test [1, 3, 5] == doL(bkt!, LIST)
    @test [1, 3, 5] == doL(bk!, LIST, binarysearch)
    @test [1, 3, 5] == doL(bk!, LIST, doublingsearch)
    @test [1, 3, 5] == doL(bk!, LIST, doublingsearchrev)

    for salgo in [binarysearch, doublingsearch, doublingsearchrev]
        n = 300
        for i in 1:n
            L = randset(n, 3)
            I = sort(intersect(L...))
            S = doL(bk!, L, salgo)
            @test I == S
        end
    end
end


@testset "Merge/Union" begin
    @test [1, 2, 3, 4, 5, 6, 7] == doT(umerge!, [[1, 2, 3, 4, 5, 6], [1, 3, 5, 7]], 1)
    @test [1, 2, 3, 4, 5, 6, 7] == doT(bkt!, [[1, 2, 3, 4, 5, 6], [1, 3, 5, 7]], 1)
    n = 300
    for i in 1:n
        L = randset(n, 5)
        I = sort(union(L...))
        #@info (i, I, bk, salgo)
        S = doT(umerge!, L, 1)
        @test I == S
        S = doT(bkt!, L, 1)
        @test I == S
    end
end

@testset "threshold" begin
    @test [1, 2, 3, 4, 5, 6, 7] == doT(umerge!, [[1, 2, 3, 4, 5, 6], [1, 3, 7], [1, 6]], 1)
    @test [1, 3, 6] == doT(umerge!, [[1, 2, 3, 4, 5, 6], [1, 3, 7], [1, 6]], 2)
    @test [1] == doT(umerge!, [[1, 2, 3, 4, 5, 6], [1, 3, 7], [1, 6]], 3)
    
    @test [1, 2, 3, 4, 5, 6, 7] == doT(bkt!, [[1, 2, 3, 4, 5, 6], [1, 3, 7], [1, 6]], 1)
    @test [1, 3, 6] == doT(bkt!, [[1, 2, 3, 4, 5, 6], [1, 3, 7], [1, 6]], 2)
    @test [1] == doT(bkt!, [[1, 2, 3, 4, 5, 6], [1, 3, 7], [1, 6]], 3)
    
    n = 3
    for i in n:n
        L = randset(n, 5)
        for t in 1:length(L)
            G = doT(umerge!, L, t) 
            @test G == doT(bkt!, L, t)
            @test G == doT(xmerge!, L, t)
        end
    end
end

