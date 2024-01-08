# This file is a part of SimilaritySearch.jl

using SimilaritySearch, LoopVectorization, LinearAlgebra, Random

struct NormalizedCosineDistanceLV{Dim} <: SemiMetric
end

function SimilaritySearch.evaluate(::NormalizedCosineDistanceLV{Dim}, u::AbstractVector{T}, v::AbstractVector{T}) where {Dim,T}
    d = zero(T)
    @turbo unroll=2 thread=1 for i in 1:Dim
        d = muladd(u[i], v[i], d)
        # d += u[i] * v[i]
    end

    one(T) - d
end

function create_database(dim=100, filled=8, n=100_000)
    X = zeros(Float32, dim, n)
    for i in 1:n
        rand!(view(X, 1:filled, i))
    end
    
    for c in eachcol(X) normalize!(c) end
    #MatrixDatabase(X), NormalizedCosineDistanceLV{dim}()
    StrideMatrixDatabase(X), NormalizedCosineDistanceLV{dim}()   # slow compilation, fast computation
end

function main()
    @info "this benchmark is intended to work with multithreading enabled julia sessions"
    db, dist = create_database(8, 8)
    k = 32
    ctx = SearchGraphContext(
                             hyperparameters_callback=OptimizeParameters(MinRecall(0.9)),
                             logger=nothing,
                             parallel_block=256
                            )
    @info "----- computing gold standard"
    GC.enable(false)
    goldsearchtime = @elapsed gI, gD = allknn(ExhaustiveSearch(; db, dist), ctx, k)
    GC.enable(true)
    @info "----- computing search graph"
    H = SearchGraph(; db, dist)
    index!(H, ctx)
    optimize_index!(H, ctx, ksearch=k)
    # prune!(RandomPruning(12), H)
    GC.enable(false)
    searchtime = @elapsed hI, hD = allknn(H, ctx, k)
    GC.enable(true)
    n = length(db)
    @info "gold:" (; n, goldsearchtime, qps=n/goldsearchtime)
    @info "searchgraph:" (; n, searchtime, qps=n / searchtime)
    @info "recall:" macrorecall(gI, hI)
    H
end

main()
