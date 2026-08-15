# This file is a part of SimilaritySearch.jl

using Test

using SimilaritySearch, Test, Base.Order
using SimilaritySearch: heapify!, heapsort!, isheap, pop_min!

@testset "heap" begin
    for k in [7, 8, 12, 15, 16, 31, 32, 67]
        X = rand(Float32, k)
        ids = UInt32.(1:k)
        
        _lt = (T, i, j) -> T[2][i] < T[2][j]
        _swap = (T, i, j) -> begin
            T[1][i], T[1][j] = T[1][j], T[1][i]
            T[2][i], T[2][j] = T[2][j], T[2][i]
        end
        T = (ids, X)
        heapify!(_lt, _swap, T, k)
        @test isheap(_lt, T, k)
        heapsort!(_lt, _swap, T, k)
        @test issorted(X, rev=false) # heapsort puts max at the end, resulting in ascending order
    end

end


@testset "KnnHeap" begin
    for k in [7, 8, 12, 15, 67]
        R = knnqueue(KnnHeap, k)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            if i % 10 == 0
                @test sort!(collect(IdDistView(R)), by=x -> x.dist) == gold
            end
            push!(gold, IdDist(i, p))
            sort!(gold, by=x -> x.dist)
            length(gold) > k && pop!(gold)

            push_item!(R, i => p)
            if i % 10 == 0
                @test sort!(collect(IdDistView(R)), by=x -> x.dist) == gold
                @test minimum(x -> x.dist, gold) == minimum(R)
                @test maximum(x -> x.dist, gold) == maximum(R)
                @test argmin(x -> x.dist, gold).id == argmin(R) || minimum(x -> x.dist, gold) == minimum(R)
                @test argmax(x -> x.dist, gold).id == argmax(R) || maximum(x -> x.dist, gold) == maximum(R)
            end
        end

        @test sortitems!(R) == gold
    end

end

@testset "XKnn" begin
    for k in [7, 8, 12, 15, 67]
        R = knnqueue(KnnSorted, k)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            @assert collect(IdDistView(R)) == gold
            push!(gold, IdDist(i, p))
            sort!(gold, by=x -> x.dist)
            length(gold) > k && pop!(gold)
            push_item!(R, i => p)
            @assert collect(IdDistView(R)) == gold

            if i % 10 == 0
                @test minimum(x -> x.dist, gold) == minimum(R)
                @test maximum(x -> x.dist, gold) == maximum(R)
                @test argmin(x -> x.dist, gold).id == argmin(R) || minimum(x -> x.dist, gold) == minimum(R)
                @test argmax(x -> x.dist, gold).id == argmax(R) || maximum(x -> x.dist, gold) == maximum(R)
                @test issorted(IdDistView(R), SimilaritySearch.DistOrder)
            end
        end

        A = collect(DistView(sortitems!(R)))
        B = collect(DistView(gold))
        @test sum(A .- B) < 1e-3

    end
end

canonical_sort(v) = sort(v, by=x -> (x.dist, x.id))  # tie-break by id: neither heapsort nor Base's default sort! is stable

@testset "RadiusSorted" begin
    for radius in Float32[0.1, 0.3, 0.6]
        R = RadiusSorted(radius)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            item = IdDist(i, p)
            accepted = push_item!(R, i => p)
            @test accepted == (p <= radius)
            p <= radius && push!(gold, item)
            @test canonical_sort(collect(IdDistView(R))) == canonical_sort(gold)
        end

        @test length(R) == length(gold)
        @test maximum(R) == radius
        @test covradius(R) == radius
        @test maxlength(R) == typemax(Int32)
        if !isempty(gold)
            g = canonical_sort(gold)
            @test nearest(R) == first(g) && frontier(R) == last(g)
        end

        reuse!(R, 0.1f0)
        @test length(R) == 0 && R.radius == 0.1f0 && isempty(R.ids)
    end
end

@testset "RadiusHeap" begin
    for radius in Float32[0.1, 0.3, 0.6]
        R = RadiusHeap(radius)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            item = IdDist(i, p)
            accepted = push_item!(R, i, p)
            @test accepted == (p <= radius)
            p <= radius && push!(gold, item)
        end

        g = canonical_sort(gold)
        @test canonical_sort(collect(IdDistView(R))) == g
        @test length(R) == length(gold)
        @test maximum(R) == radius
        @test covradius(R) == radius
        if !isempty(g)
            @test nearest(R) == first(g) && frontier(R) == last(g)
        end

        reuse!(R, 0.1f0)
        @test length(R) == 0 && R.radius == 0.1f0 && isempty(R.ids)
    end
end

@testset "XKnn pop ops" begin
    for k in [7, 12, 31]
        R = knnqueue(KnnSorted, k)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            @assert collect(IdDistView(R)) == gold
            push!(gold, IdDist(i, p))
            sort!(gold, by=x -> x.dist)
            length(gold) > k && pop!(gold)

            push_item!(R, i => p)
            @assert collect(IdDistView(R)) == gold

            if i % 10 == 7
                p = pop_min!(R)
                @test p == popfirst!(gold)
                p = pop_max!(R)
                @test p == pop!(gold)
            end

            if i % 10 == 0
                @test minimum(x -> x.dist, gold) == minimum(R)
                @test maximum(x -> x.dist, gold) == maximum(R)
                @test issorted(IdDistView(R), SimilaritySearch.DistOrder)
            end
        end

        A = collect(DistView(sortitems!(R)))
        B = collect(DistView(gold))
        @test sum(A .- B) < 1e-3
    end

end
