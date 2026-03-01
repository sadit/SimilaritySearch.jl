# This file is a part of SimilaritySearch.jl

using Test

using SimilaritySearch, Test, Base.Order
using SimilaritySearch: heapify!, heapsort!, isheap, pop_min!

@testset "heap" begin
    for k in [7, 8, 12, 15, 16, 31, 32, 67]
        X = rand(Float32, k)
        heapify!(Forward, X)
        @test isheap(Forward, X)
        heapsort!(Forward, X)
        @test issorted(X)
    end

end


@testset "KnnHeap" begin
    for k in [7, 8, 12, 15, 67]
        R = knnqueue(KnnHeap, k)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            @test sort!(collect(viewitems(R)), by=x -> x.dist) == gold
            # i > 7 && (p *= maximum(R))
            push!(gold, IdDist(i, p))
            sort!(gold, by=x -> x.dist)
            length(gold) > k && pop!(gold)

            #@show "======================="
            #@show "PRE", gold, sort!(collect(viewitems(R)), by=x->x.dist), i => p
            #@show R.min, R.len, R.maxlen
            push_item!(R, i => p)
            #@show "POS", gold, sort!(collect(viewitems(R)), by=x->x.dist), i => p
            #@show R.min, R.len, R.maxlen
            @test sort!(collect(viewitems(R)), by=x -> x.dist) == gold

            @test minimum(x -> x.dist, gold) == minimum(R)
            @test maximum(x -> x.dist, gold) == maximum(R)
            @test argmin(x -> x.dist, gold).id == argmin(R) || minimum(x -> x.dist, gold) == minimum(R)
            @test argmax(x -> x.dist, gold).id == argmax(R) || maximum(x -> x.dist, gold) == maximum(R)
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
            # i > 7 && (p *= maximum(R))
            @assert collect(viewitems(R)) == gold
            push!(gold, IdDist(i, p))
            sort!(gold, by=x -> x.dist)
            length(gold) > k && pop!(gold)
            #@show "======================="
            #@show "PRE" gold, collect(viewitems(R)), i => p
            push_item!(R, i => p)
            #@show "POS" gold, collect(viewitems(R)), i => p
            @assert collect(viewitems(R)) == gold

            @test minimum(x -> x.dist, gold) == minimum(R)
            @test maximum(x -> x.dist, gold) == maximum(R)
            @test argmin(x -> x.dist, gold).id == argmin(R) || minimum(x -> x.dist, gold) == minimum(R)
            @test argmax(x -> x.dist, gold).id == argmax(R) || maximum(x -> x.dist, gold) == maximum(R)
            #@test issorted(viewitems(R), SimilaritySearch.RevDistOrder)
            @test issorted(viewitems(R), SimilaritySearch.DistOrder)
        end

        #@show DistView(sortitems!(R))
        A = collect(DistView(sortitems!(R)))
        B = collect(DistView(gold))
        @test sum(A .- B) < 1e-3

    end
end

@testset "XKnn pop ops" begin
    for k in [7, 12, 31]
        R = knnqueue(KnnSorted, k)
        gold = IdDist[]

        for i in Int32(1):Int32(10^3)
            p = rand(Float32)
            # i > 7 && (p *= maximum(R))
            @assert collect(viewitems(R)) == gold
            push!(gold, IdDist(i, p))
            sort!(gold, by=x -> x.dist)
            length(gold) > k && pop!(gold)

            push_item!(R, i => p)
            @assert collect(viewitems(R)) == gold

            if i % 10 == 7
                p = pop_min!(R)
                @test p == popfirst!(gold)
                p = pop_max!(R)
                @test p == pop!(gold)
            end

            @test minimum(x -> x.dist, gold) == minimum(R)
            @test maximum(x -> x.dist, gold) == maximum(R)
            #@test issorted(viewitems(R), SimilaritySearch.RevDistOrder)
            @test issorted(viewitems(R), SimilaritySearch.DistOrder)
            # i == 3 && break
        end

        A = collect(DistView(sortitems!(R)))
        B = collect(DistView(gold))
        @test sum(A .- B) < 1e-3
    end

end
