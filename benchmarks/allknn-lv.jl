
using SimilaritySearch, LoopVectorization, StrideArrays, LinearAlgebra, Random

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

function create_database(dim=100, n=100_000, filled=8)
    X = zeros(Float32, dim, n)
    for i in 1:n
        rand!(view(X, 1:filled, i))
    end
    
    for c in eachcol(X) normalize!(c) end
    # MatrixDatabase(X), NormalizedCosineDistanceLV{dim}()
    MatrixDatabase(StrideArray(X, StaticInt.(size(X)))), NormalizedCosineDistanceLV{dim}()   # slow compilation, fast computation
    # MatrixDatabase(StrideArray(X, size(X))), NormalizedCosineDistanceLV{8}()
    #MatrixDatabase(X), NormalizedCosineDistance()   # fast compilation, a bit slower
end

function main()
    @info "this benchmark is intended to work with multithreading enabled julia sessions"
    db, dist = create_database()
    k = 32
    @info "----- computing gold standard"
    goldsearchtime = @elapsed gI, gD = allknn(ExhaustiveSearch(; db, dist), k)
    @info "----- computing search graph"
    H = SearchGraph(; db, dist, verbose=false)
    index!(H; parallel_block=256)
    optimize!(H, MinRecall(0.9), ksearch=k)
    # prune!(RandomPruning(12), H)
    searchtime = @elapsed hI, hD = allknn(H, k; minbatch=32)
    n = length(db)
    @info "gold:" (; n, goldsearchtime, qps=n/goldsearchtime)
    @info "searchgraph:" (; n, searchtime, qps=n / searchtime)
    @info "recall:" macrorecall(gI, hI)
    H
end

#GC.enable(false)
main()
#GC.enable(true)
