# This file is part of SimilaritySearch.jl

using SimilaritySearch, SimilaritySearch.InvertedFiles, LinearAlgebra
using SimilaritySearch: Dist, evaluate
using Test
using Random
Random.seed!(0)

@testset "WeightedInvertedFile" begin
    A = MatrixDatabase(normalize!(rand(300, 1000)))
    B = VectorDatabase([Dict(enumerate(a)) for a in A])

    # testing with Vector container (map=nothing)
    ectx = GenericContext()
    ctx = InvertedFileContext()
    I = append_items!(WeightedInvertedFile(300), ctx, B)

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(Dist.NormCosine(), A), ectx, A[qid], knnqueue(KnnSorted, k))
        b = search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        @test recallscore(a, b) == 1.0
        #if i == 1
        #  @test_call search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        #end
    end

    # testing with Dict container (map != nothing)
    I = append_items!(WeightedInvertedFile(300), ctx, B)

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(Dist.NormCosine(), A), ectx, A[qid], knnqueue(KnnSorted, k))
        b = search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        @test recallscore(a, b) == 1.0
        #if i == 1
        #  @test_call search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        #end
    end

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(Dist.NormCosine(), A), ectx, A[qid], knnqueue(KnnSorted, k))
        b = search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        @test recallscore(a, b) == 1.0
        #if i == 1
        #  @test_call search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        #end
    end

    ## working on sparse data
    # increasing sparsity of the arrays
    for A_ in A
        t = partialsort(A_, 7, rev=true)
        for i in eachindex(A_)
            A_[i] = A_[i] < t ? 0.0 : A_[i]
        end
        normalize!(A_)
    end

    create_sparse(A_) = Dict([i => a for (i, a) in enumerate(A_) if a > 0.0])

    #B = VectorDatabase([create_sparse(A_) for A_ in A])
    B = VectorDatabase(A)
    I = append_items!(WeightedInvertedFile(300), ctx, B)
    k = 1  # the aggresive cut of the attributes need a small k
    @test length(I) == length(B)
    for i in 1:10
        #@info i
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(Dist.NormCosine(), A), ectx, A[qid], knnqueue(KnnSorted, k))
        b = search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        @test recallscore(a, b) == 1.0
        #@show recallscore(a, b)
        #if i == 1
        #  @test_call search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        #end
    end

    I = WeightedInvertedFile(300)
    @test length(I) == 0
    append_items!(I, ctx, B)
    @test length(I) == length(B)
    k = 1  # the aggresive cut of the attributes need a small k
    for i in 1:10
        # @info i
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(Dist.NormCosine(), A), ectx, A[qid], knnqueue(KnnSorted, k))
        b = search(I, ctx, B[qid], knnqueue(KnnSorted, k))
        @test recallscore(a, b) == 1.0
        #@show recallscore(a, b)
    end

    ak = allknn(ExhaustiveSearch(Dist.NormCosine(), B), ectx, 3)
    @test 1.0 == macrorecall(ak, searchbatch(I, ctx, B, 3))

    #=@testset "saveindex and loadindex WeightedInvertedFile" begin
        tmpfile = tempname()
        @info "--- load and save!!!"

        saveindex(tmpfile, I; meta=[1, 2, 4, 8], store_db=false)
        let
            G, meta = loadindex(tmpfile, database(I); staticgraph=true)
            @test meta == [1, 2, 4, 8]
            @test G.adj isa StaticAdjacencyList
            @test 1.0 == macrorecall(ak, allknn(G, ctx, 3))
        end
    end=#
end

@testset "BinaryInvertedFile" begin
    vocsize = 128
    n = 10_000
    m = 100
    len = 10
    k = 10
    db = VectorDatabase([sort!(unique(rand(1:vocsize, len))) for i in 1:n])
    queries = VectorDatabase([sort!(unique(rand(1:vocsize, len))) for i in 1:m])
    ectx = GenericContext()
    ctx = InvertedFileContext()

    #for dist in [JaccardDistance(), DiceDistance(), CosineDistanceSet(), IntersectionDissimilarity()]
    for dist in [Dist.Sets.Jaccard()]
        S = ExhaustiveSearch(dist, db)
        gold = searchbatch(S, ectx, queries, k)

        IF = BinaryInvertedFile(vocsize, dist)
        append_items!(IF, ctx, db)
        knns = searchbatch(IF, ctx, queries, k)
        ctx = getcontext(IF)
        @time search(IF, ctx, queries[1], knnqueue(KnnSorted, k))
        @time search(IF, ctx, queries[2], knnqueue(KnnSorted, k))
        #@test_call search(IF, ctx, queries[2], knnqueue(KnnSorted, k))
        recall = macrorecall(gold, knns)
        @show dist, recall
        @test recall > 0.95  # sets can be tricky since we can expect many similar distances
        err = 0.0
        for i in 1:m
            d = evaluate(Dist.L2(), collect(Float32, DistView(gold[:, i])), collect(Float32, DistView(knns[:, i])))
            err += d
            if d > 0.1
                @info dist, i, gold[:, i], knns[:, i]
                @info dist, i, queries[i]
            end
        end
        @show dist, err
        @test err < 0.01  # acc. floating point errors

        #=@testset "saveindex and loadindex BinaryInvertedFile" begin
            tmpfile = tempname()
            @info "--- load and save!!!"
            saveindex(tmpfile, IF; meta=[1, 2, 4, 8], store_db=false)
            let
                G, meta = loadindex(tmpfile, database(IF); staticgraph=true)
                @test meta == [1, 2, 4, 8]
                @test G.adj isa StaticAdjacencyList
                knns = searchbatch(G, ctx, queries, k)
                recall = macrorecall(gold, knns)
                @test recall > 0.95
            end
        end=#

    end
end
