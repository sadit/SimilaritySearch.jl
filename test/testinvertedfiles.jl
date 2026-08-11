# This file is part of SimilaritySearch.jl

using SimilaritySearch, SimilaritySearch.InvertedFiles, LinearAlgebra, SparseArrays
using SimilaritySearch: Dist, evaluate
using Test
using Random
Random.seed!(0)

@testset "InvertedFile with Dist.NormCosine()" begin
    @test !SimilaritySearch.InvertedFiles.has_exact_fastpath(Dist.NormCosine())

    A = MatrixDatabase(normalize!(rand(300, 1000)))
    B = VectorDatabase([sparse(a) for a in A])

    ectx = GenericContext()
    ctx = InvertedFileContext()
    I = append_items!(InvertedFile(300, Dist.NormCosine()), ctx, B)

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

    I = append_items!(InvertedFile(300, Dist.NormCosine()), ctx, B)

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

    B = VectorDatabase([sparse(a) for a in A])
    I = append_items!(InvertedFile(300, Dist.NormCosine()), ctx, B)
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

    I = InvertedFile(300, Dist.NormCosine())
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

    ak_ids, ak_dists = allknn(ExhaustiveSearch(Dist.NormCosine(), B), ectx, 3)
    I_knns_ids, I_knns_dists = searchbatch(I, ctx, B, 3)
    @test 1.0 == macrorecall(ak_ids, I_knns_ids)

    #=@testset "saveindex and loadindex InvertedFile" begin
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

@testset "InvertedFile" begin
    vocsize = 128
    n = 2_000
    m = 30
    len = 10
    k = 10
    db = VectorDatabase([sort!(unique(rand(1:vocsize, len))) for i in 1:n])
    queries = VectorDatabase([sort!(unique(rand(1:vocsize, len))) for i in 1:m])
    ectx = GenericContext()
    ctx = InvertedFileContext()

    # exact fast-path set metrics: score is computed purely from intersection size + set sizes,
    # so recall/error should match ExhaustiveSearch tightly (no rerank! needed for these).
    for dist in [Dist.Sets.Jaccard(), Dist.Sets.RogersTanimoto(vocsize)]
        @test SimilaritySearch.InvertedFiles.has_exact_fastpath(dist)

        S = ExhaustiveSearch(dist, db)
        gold_ids, gold_dists = searchbatch(S, ectx, queries, k)

        IF = InvertedFile(vocsize, dist)
        append_items!(IF, ctx, db)
        @test length(database(IF)) == length(db)
        knns_ids, knns_dists = searchbatch(IF, ctx, queries, k)
        ctx = getcontext(IF)
        @time search(IF, ctx, queries[1], knnqueue(KnnSorted, k))
        @time search(IF, ctx, queries[2], knnqueue(KnnSorted, k))
        #@test_call search(IF, ctx, queries[2], knnqueue(KnnSorted, k))
        recall = macrorecall(gold_ids, knns_ids)
        @show dist, recall
        @test recall > 0.95  # sets can be tricky since we can expect many similar distances
        err = 0.0
        for i in 1:m
            d = evaluate(Dist.L2(), gold_dists[:, i], knns_dists[:, i])
            err += d
            if d > 0.1
                @info dist, i, gold_dists[:, i], knns_dists[:, i]
                @info dist, i, queries[i]
            end
        end
        @show dist, err
        @test err < 0.01  # acc. floating point errors
    end

    @testset "generic distance, direct-evaluate fallback path" begin
        # a distance with no closed-form set_distance_evaluate case; search must evaluate `dist`
        # directly against `db` for every merge candidate (see `FallbackInvFileOutput`)
        struct SizeDiffDist <: Dist.SemiMetric end
        SimilaritySearch.evaluate(::SizeDiffDist, a, b)::Float32 = abs(Float32(length(a)) - Float32(length(b)))

        dist = SizeDiffDist()
        @test !SimilaritySearch.InvertedFiles.has_exact_fastpath(dist)

        IF = InvertedFile(vocsize, dist)
        append_items!(IF, ctx, db)
        q = queries[1]

        costs = Int[]
        for t in (1, 2)
            ctx_t = getcontext(IF)
            res = search(IF, ctx_t, q, knnqueue(KnnSorted, k); t)
            @test length(res) > 0

            for it in viewitems(res)
                @test evaluate(dist, database(IF)[it.id], q) ≈ it.dist
            end
            dists = [it.dist for it in viewitems(res)]
            @test issorted(dists)
            # every onmatch! call pays exactly one real `evaluate()` call, so at least one per
            # surviving item, but possibly more (candidates evaluated then evicted from `res`)
            cost = sum(ctx_t.costdists)
            @test cost >= length(res)
            push!(costs, cost)
        end
        # raising `t` tightens the merge-agreement requirement, so it must not increase the
        # number of real `evaluate()` calls (it's the fallback path's cost-control knob)
        @test costs[2] <= costs[1]
    end

    @testset "sparseiterator(dist, obj) is overloadable per (DistType, ObjType)" begin
        struct TaggedWeightDist <: Dist.SemiMetric end
        SimilaritySearch.InvertedFiles.sparseiterator(::TaggedWeightDist, obj::AbstractVector{<:Integer}) =
            ((u, 7) for u in obj)

        default_iter = collect(SimilaritySearch.InvertedFiles.sparseiterator(db[1]))
        tagged_iter = collect(SimilaritySearch.InvertedFiles.sparseiterator(TaggedWeightDist(), db[1]))
        @test all(w == 7 for (_, w) in tagged_iter)
        @test any(w != 7 for (_, w) in default_iter)
        @test [id for (id, _) in default_iter] == [id for (id, _) in tagged_iter]
    end
end
