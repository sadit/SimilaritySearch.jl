# This file is a part of SimilaritySearch.jl

export recallscore, macrorecall, matcherror, macromatcherror, perqueryscores, bootstrapscore, BootstrapScore

using Statistics: quantile, std
using Random: AbstractRNG, default_rng

"""
    recallscore(gold, res) -> Float64

Computes the recall score of a single result set `res` against its gold standard `gold`, i.e., the
fraction of the identifiers in `gold` that also appear in `res`. Both `gold` and `res` can be a `Set`,
an `AbstractVector{IdDist}`, an `AbstractVector{<:Integer}`, or an `AbstractKnnQueue` object.

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
idset(res::AbstractKnnQueue) = Set{UInt32}(IdView(res))

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
E = ExhaustiveSearch(dist, X)
ctx = getcontext(E)

gold = searchbatch(E, ctx, X, 8)
G = SearchGraph(dist, X)
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
of per-query result sets (each element can be a `Set`, an `AbstractKnnQueue` object, or a vector of identifiers)
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

"""
    matcherror(golddist::AbstractVector{Float32}, res::AbstractKnnQueue; exponent=1, maxdeviation=1, spreadfloor=1f-2) -> Float64
    matcherror(golddist, res, err::MaxMatchError) -> Float64

Per-query MatchError (see [`MaxMatchError`](@ref)): the distances actually returned in `res`
against the exact gold distances `golddist`, rank by rank, each position's excess measured in
units of the gold neighborhood's own spread. For `k' = length(golddist)` gold distances
`d*_1 ≤ … ≤ d*_k'` and the `r` returned distances `d_1 ≤ … ≤ d_r`:

    spread      = d*_k' − min(d*_1, d_1) + spreadfloor + ε
    deviation_i = min(max(0, d_i − d*_i) / spread, maxdeviation)   for i ≤ r
    deviation_i = maxdeviation                                     for i > r   (a missing position)
    matcherror  = mean(deviation_i ^ exponent  for i in 1:k')

- `exponent`: `1` averages the deviations (an MAE), `2` squares them first, which forgives
  small ones and amplifies large ones.
- `maxdeviation`: the most one position can cost, and what a missing position costs: a
  returned neighbor farther than `maxdeviation` spreads beyond its gold counterpart is as bad
  as no neighbor at all. The score is therefore in `[0, maxdeviation ^ exponent]`, which is
  what keeps its mean over queries meaningful: measured on SISAP 2025 `ccnews`, without this
  cap ten queries whose gold neighbors were all exact duplicates at distance 0 (spread 0, the
  search returning items at distance 1.4, 143 per position) made 85% of the mean over 10,500
  queries, and the mean moved by its own size from one tuning run to the next.
- `spreadfloor`: added to the spread so a gold neighborhood of tied distances (routine on
  real data with near-duplicates) does not divide by `≈ε`; pick it on the scale of the
  distance (`1f-2` suits a cosine-family distance in `[0, 2]`; `Hamming` over bits wants
  about `1f0`).

[`macromatcherror`](@ref) is its mean over the queries, and the third form takes the three
parameters from a [`MaxMatchError`](@ref).
"""
function matcherror(golddist::AbstractVector{Float32}, res::AbstractKnnQueue; exponent::Real=1, maxdeviation::Real=1, spreadfloor::Real=1f-2)::Float64
    sortitems!(res)
    _matcherror(golddist, DistView(res), length(res), exponent, maxdeviation, spreadfloor)
end

"""
    matcherror(golddist::AbstractVector{Float32}, res::BallKnn; exponent=1, maxdeviation=1, spreadfloor=1f-2) -> Float64

MatchError of a radius-bounded search: scores only the items `res` holds *within its radius*,
never its navigation reserve (see [`BallKnn`](@ref)), against the true ball's distances. Each ball
member the search did not reach costs `maxdeviation`, which is what makes this the radius
counterpart of recall -- and why [`MaxMatchError`](@ref) is the only `ErrorFunction` that
transfers to radius queries: [`MinRecall`](@ref) goes through `macrorecall`, which divides by the
gold set's size, and a small radius routinely produces queries whose true ball is empty.
"""
function matcherror(golddist::AbstractVector{Float32}, res::BallKnn; exponent::Real=1, maxdeviation::Real=1, spreadfloor::Real=1f-2)::Float64
    _matcherror(golddist, DistView(res), ninside(res), exponent, maxdeviation, spreadfloor)
end

function _matcherror(golddist::AbstractVector{Float32}, returned, nreturned::Integer, exponent::Real, maxdeviation::Real, spreadfloor::Real)::Float64
    ngold = length(golddist)
    ngold == 0 && return 0.0
    nearest = nreturned > 0 ? min(golddist[1], @inbounds(returned[1])) : golddist[1]
    spread = golddist[ngold] - nearest + spreadfloor + eps(Float32)

    total = 0.0
    @inbounds for i in 1:ngold
        deviation = i <= nreturned ? min(max(0f0, returned[i] - golddist[i]) / spread, maxdeviation) : maxdeviation
        total += deviation^exponent
    end

    total / ngold
end


_ncolumns(x::AbstractMatrix) = size(x, 2)
_ncolumns(x::AbstractVector) = length(x)
_column(x::AbstractMatrix, i, k) = view(x, 1:(k === nothing ? size(x, 1) : Int(k)), i)
_column(x::AbstractVector, i, k) = x[i]

"""
    macromatcherror(golddists, reslist; exponent=1, maxdeviation=1, spreadfloor=1f-2) -> Float64
    macromatcherror(golddists, reslist, err::MaxMatchError) -> Float64

The mean of the per-query [`matcherror`](@ref) over the queries: the macro MatchError, the
distance-based counterpart of [`macrorecall`](@ref), and what [`MaxMatchError`](@ref) scores
a configuration by (the second form takes the parameters from `err`). `golddists` holds each query's exact gold distances in ascending
order, as a vector with one vector per query or as a `(k, n)` matrix (the second output of
[`searchbatch`](@ref) over an exact index), and `reslist` the result queues to score, one per
query. `0` is a perfect match, `maxdeviation ^ exponent` the worst.

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10^3))
Q = MatrixDatabase(rand(Float32, 8, 100))
E = ExhaustiveSearch(Dist.SqL2(), X)
goldI, goldD = searchbatch(E, GenericContext(), Q, 10)
G = SearchGraph(Dist.SqL2(), X); ctx = SearchGraphContext(); index!(G, ctx)
knns = [search(G, ctx, Q[i], knnqueue(KnnSorted, 10)) for i in 1:length(Q)]
macromatcherror(goldD, knns)          # exponent 1, maxdeviation 1: the MaxMatchError() defaults
```
"""
function macromatcherror(golddists, reslist; exponent::Real=1, maxdeviation::Real=1, spreadfloor::Real=1f-2)::Float64
    n = _ncolumns(reslist)
    _ncolumns(golddists) == n || throw(DimensionMismatch("macromatcherror: $(_ncolumns(golddists)) gold queries against $n results"))
    s = 0.0
    for i in 1:n
        s += matcherror(_column(golddists, i, nothing), _column(reslist, i, nothing); exponent, maxdeviation, spreadfloor)
    end
    s / n
end

"""
    perqueryscores(score, gold, res; k=nothing) -> Vector{Float64}

`score(gold_i, res_i)` for every query `i`: the vector a macro score is the mean of, and
what [`bootstrapscore`](@ref) resamples. `gold` and `res` are each either a matrix with one
column per query, cut to its first `k` rows when `k` is given (as [`macrorecall`](@ref)
does), or a vector with one entry per query; `score` is any two-argument function --
[`recallscore`](@ref), `(g, r) -> matcherror(g, r; exponent=2)`, or a user's.

```julia
perqueryscores(recallscore, goldI, resI)                       # what macrorecall averages
perqueryscores(matcherror, goldD, knns)                        # what macromatcherror averages
```
"""
function perqueryscores(score, gold, res; k::Union{Nothing,Integer}=nothing)
    n = _ncolumns(gold)
    _ncolumns(res) == n || throw(DimensionMismatch("perqueryscores: $n gold queries against $(_ncolumns(res)) results"))
    [Float64(score(_column(gold, i, k), _column(res, i, k))) for i in 1:n]
end

"""
    BootstrapScore

What [`bootstrapscore`](@ref) returns: `mean`, the macro score itself (the mean of
`perquery`); `std`, the standard deviation of the resampled macro scores; `lo` and `hi`, the
percentile confidence interval at `level`; `samples`, one macro score per resample; and
`perquery`, the per-query scores they were drawn from.
"""
struct BootstrapScore
    mean::Float64
    std::Float64
    lo::Float64
    hi::Float64
    level::Float64
    samples::Vector{Float64}
    perquery::Vector{Float64}
end

function Base.show(io::IO, b::BootstrapScore)
    r(x) = round(x; digits=4)
    print(io, "BootstrapScore(", r(b.mean), " ± ", r(b.std), ", ", round(Int, 100 * b.level), "% [", r(b.lo), ", ", r(b.hi), "], ",
          length(b.perquery), " queries, ", length(b.samples), " resamples)")
end

"""
    bootstrapscore(perquery::AbstractVector{<:Real}; nboot=1000, level=0.95, rng=Random.default_rng()) -> BootstrapScore
    bootstrapscore(score, gold, res; k=nothing, nboot=1000, level=0.95, rng=Random.default_rng()) -> BootstrapScore

The distribution of a macro score under resampling of the queries. A macro score
([`macrorecall`](@ref), [`macromatcherror`](@ref)) is the mean of a per-query score, and a
single number says nothing about how much it would move with another sample of queries;
this draws `length(perquery)` queries with replacement `nboot` times, takes the mean of each
draw, and returns the mean, the standard deviation, and the percentile interval at `level` of
those means (see [`BootstrapScore`](@ref)). The per-query scores are computed once, by
[`perqueryscores`](@ref) in the second form, and only that vector is resampled, so `nboot`
costs nothing next to the searches that produced `res`; `rng` makes the draws reproducible.

Two results on the same queries are compared **paired**: bootstrap the per-query
differences, so each draw takes the same queries from both, and read whether the interval
excludes zero:

```julia
a = perqueryscores(recallscore, goldI, resA)
b = perqueryscores(recallscore, goldI, resB)
d = bootstrapscore(a .- b; nboot=10_000)      # d.lo > 0: A is better on these queries at that level
```

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10^3))
Q = MatrixDatabase(rand(Float32, 8, 100))
E = ExhaustiveSearch(Dist.SqL2(), X)
goldI, goldD = searchbatch(E, GenericContext(), Q, 10)
G = SearchGraph(Dist.SqL2(), X); ctx = SearchGraphContext(); index!(G, ctx)
resI, _ = searchbatch(G, ctx, Q, 10)
bootstrapscore(recallscore, goldI, resI)                       # BootstrapScore(0.97 ± 0.0056, 95% [0.959, 0.981], 100 queries, 1000 resamples)
knns = [search(G, ctx, Q[i], knnqueue(KnnSorted, 10)) for i in 1:length(Q)]
bootstrapscore((g, r) -> matcherror(g, r, MaxMatchError()), goldD, knns)
```
"""
function bootstrapscore(perquery::AbstractVector{<:Real}; nboot::Integer=1000, level::Real=0.95, rng::AbstractRNG=default_rng())
    n = length(perquery)
    n > 0 || throw(ArgumentError("bootstrapscore: no queries to resample"))
    nboot > 1 || throw(ArgumentError("bootstrapscore: nboot=$nboot must be at least 2"))
    0 < level < 1 || throw(ArgumentError("bootstrapscore: level=$level must be in (0, 1)"))
    pq = Vector{Float64}(perquery)
    samples = Vector{Float64}(undef, nboot)
    @inbounds for b in 1:nboot
        s = 0.0
        for _ in 1:n
            s += pq[rand(rng, 1:n)]
        end
        samples[b] = s / n
    end
    α = (1 - level) / 2
    lo, hi = quantile(samples, [α, 1 - α])
    BootstrapScore(sum(pq) / n, std(samples), lo, hi, Float64(level), samples, pq)
end

bootstrapscore(score, gold, res; k::Union{Nothing,Integer}=nothing, kwargs...) =
    bootstrapscore(perqueryscores(score, gold, res; k); kwargs...)
