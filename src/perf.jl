# This file is a part of SimilaritySearch.jl

export recallscore, macrorecall

"""
    recallscore(gold, res) -> Float64

Computes the recall score of a single result set `res` against its gold standard `gold`, i.e., the
fraction of the identifiers in `gold` that also appear in `res`. Both `gold` and `res` can be a `Set`,
an `AbstractVector{IdDist}`, an `AbstractVector{<:Integer}`, or an `AbstractKnn` object.

# Arguments
- `gold`: the gold standard (exact) result set
- `res`: the result set to be evaluated

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 8, 10^3))
E = ExhaustiveSearch(; dist, db=X)
ctx = getcontext(E)

gold = searchbatch(E, ctx, X, 8)
res = searchbatch(E, ctx, X, 8)  # here identical to gold, just for illustration
recallscore(view(gold, :, 1), view(res, :, 1))  # 1.0
```
"""
function recallscore(gold, res)::Float64
    length(intersect(idset(gold), idset(res))) / length(gold)
end

idset(a::Set) = a
idset(a::AbstractVector{<:Integer}) = Set{UInt32}(a)
idset(res::AbstractKnn) = Set{UInt32}(IdView(res))

"""
    macrorecall(goldI::AbstractMatrix, resI::AbstractMatrix, k::Integer=size(goldI, 1)) -> Float64

Computes the macro recall score, i.e., the average of the per-query [`recallscore`](@ref), using `goldI` as
the gold standard and `resI` as the predictions to be evaluated; both are expected to be matrices of
identifiers (e.g., `IdDist` or integers) with one column per query. If `k` is given, then each column is
cut to its first `k` entries before scoring.

# Arguments
- `goldI`: a `(k, n)` matrix with the gold standard (exact) result of `n` queries
- `resI`: a `(k, n)` matrix with the result to be evaluated of the same `n` queries
- `k`: the number of neighbors (per column) to consider; defaults to `size(goldI, 1)`

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 8, 10^3))
E = ExhaustiveSearch(; dist, db=X)
ctx = getcontext(E)

gold = searchbatch(E, ctx, X, 8)
G = SearchGraph(; dist, db=X)
gctx = getcontext(G)
index!(G, gctx)
res = allknn(G, gctx, 8)

macrorecall(gold, res)  # macro recall of the approximate index against the exact gold standard
```
"""
function macrorecall(goldI::AbstractMatrix, resI::AbstractMatrix, k::Integer=size(goldI, 1))::Float64
    n = size(goldI, 2)
    s = 0.0
    for i in 1:n
        s += recallscore(view(goldI, 1:k, i), view(resI, 1:k, i))
    end

    s / n
end

"""
    macrorecall(goldlist::AbstractVector, reslist::AbstractVector) -> Float64

Computes the macro recall score, i.e., the average of the per-query [`recallscore`](@ref), using vectors
of per-query result sets (each element can be a `Set`, an `AbstractKnn` object, or a vector of identifiers)
instead of matrices.

# Arguments
- `goldlist`: a vector with one gold-standard result set per query
- `reslist`: a vector with one result set (to be evaluated) per query, `length(reslist) == length(goldlist)`

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 8, 200))
E = ExhaustiveSearch(; dist, db=X)
ctx = getcontext(E)

knns = searchbatch(E, ctx, X, 8)
goldlist = [Set(collect(IdView(view(knns, :, i)))) for i in 1:length(X)]
reslist = goldlist  # here identical to gold, just for illustration
macrorecall(goldlist, reslist)  # 1.0
```
"""
function macrorecall(goldlist::AbstractVector, reslist::AbstractVector)::Float64
    @assert length(goldlist) == length(reslist) "$(length(goldlist)) == $(length(reslist))"
    s = 0.0
    n = length(goldlist)
    for i in 1:n
        g = goldlist[i]
        r = reslist[i]
        s += recallscore(g, r)
    end

    s / n
end
