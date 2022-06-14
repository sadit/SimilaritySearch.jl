
using SimilaritySearch, LoopVectorization, StrideArrays, LinearAlgebra

struct NormalizedCosineDistanceLV{Dim} <: SemiMetric
end

function SimilaritySearch.evaluate(::NormalizedCosineDistanceLV{Dim}, u::AbstractVector{T}, v::AbstractVector{T}) where {Dim,T}
    d = zero(T)
    @turbo inline=true unroll=2 thread=1 for i in 1:Dim
        d = fma(u[i], v[i], d)
    end

    one(T) - d
end

function create_database()
    X = rand(Float32, 8, 100_000)
    for c in eachcol(X) normalize!(c) end
    MatrixDatabase(StrideArray(X, StaticInt.(size(X))))
end

function main()
    @info "this benchmark is intended to work with multithreading enabled julia sessions"
    db = create_database()
    dist = NormalizedCosineDistanceLV{8}()
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
