# This file is a part of SimilaritySearch.jl

using SimilaritySearch, FixedSizeArrays, LoopVectorization, LinearAlgebra, Random

struct NormalizedCosineDistanceLV{Dim} <: SemiMetric
end

function SimilaritySearch.evaluate(::NormalizedCosineDistanceLV{Dim}, u::AbstractVector{T}, v::AbstractVector{T}) where {Dim,T}
    d = zero(T)
    @turbo unroll = 4 thread = 1 for i in 1:Dim
        d = muladd(u[i], v[i], d)
        # d += u[i] * v[i]
    end

    one(T) - d
end

function create_database(dim, n)
    rng = Xoshiro(n)
    #X = Matrix{Float32}(undef, dim, n)
    X = FixedSizeMatrix{Float32}(undef, dim, n)
    rand!(rng, X)
    #=rand!(rng, X)
    X = zeros(Float32, dim, n)
    for i in 1:n
        rand!(view(X, 1:filled, i))
    end=#

    for c in eachcol(X)
        normalize!(c)
    end
    #MatrixDatabase(X), NormalizedCosineDistanceLV{dim}()
    MatrixDatabase(X), NormalizedCosineDistanceLV{dim}()   # slow compilation, fast computation
end

#using SimilaritySearch, FixedSizeArrays, LinearAlgebra, Random


function main(dim, n, k)
    @info "this benchmark is intended to work with multithreading enabled julia sessions"
    db, dist = create_database(dim, n)
    @info "----- computing gold standard"
    ctx = SearchGraphContext(
        hyperparameters_callback=OptimizeParameters(MinRecall(0.99)),
        parallel_block=1024
    )
    goldsearchtime = @elapsed gold_knns = allknn(ExhaustiveSearch(; db, dist), ctx, k)
    @info "----- computing search graph with k=$k"
    H = SearchGraph(; db, dist)
    index!(H, ctx)
    optimize_index!(H, ctx, ksearch=k)
    GC.enable(false)
    searchtime = @elapsed knns = allknn(H, ctx, k; sort=false, progress=nothing)
    GC.enable(true)
    n = length(db)
    @info "gold:" (; n, goldsearchtime, qps=n / goldsearchtime)
    @info "searchgraph:" (; n, searchtime, qps=n / searchtime)
    @info "recall:" macrorecall(gold_knns, knns)
end

main(8, 10^3, 8)
main(8, 10^5, 8)
