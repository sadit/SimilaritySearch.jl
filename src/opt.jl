# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize_index!, MinRecall, OptRadius, ParetoRecall, ParetoRadius, MaxMatchError

"""
    abstract type ErrorFunction end

Abstract type for the optimization goals (`kind` argument) accepted by [`optimize_index!`](@ref).
It determines how candidate hyperparameter configurations are scored/compared while
autotuning the index. Concrete subtypes are [`MinRecall`](@ref), [`OptRadius`](@ref),
[`ParetoRecall`](@ref), [`ParetoRadius`](@ref), and [`MaxMatchError`](@ref).
"""
abstract type ErrorFunction end

"""
    MinRecall(; minrecall=0.9f0) <: ErrorFunction

Optimization goal that favors the fastest configuration among those achieving at least
`minrecall` recall (measured against a gold standard computed with exhaustive search).

# Keyword Arguments
- `minrecall`: minimum recall (0-1) required to be considered as fast as possible.

# Examples

```julia
optimize_index!(index, ctx, MinRecall(0.95))
```
"""
@kwdef struct MinRecall <: ErrorFunction
    minrecall::Float32 = 0.9f0
end

"""
    OptRadius(; tol=0.1) <: ErrorFunction

Optimization goal that favors the fastest configuration among those whose search radius
falls within a `tol`-sized tolerance band, without relying on a computed gold standard.

# Keyword Arguments
- `tol`: relative tolerance used to bucket configurations by their achieved search radius.

# Examples

```julia
optimize_index!(index, ctx, OptRadius(; tol=0.05))
```
"""
@kwdef struct OptRadius <: ErrorFunction
    tol::Float32 = 0.1
end

"""
    ParetoRecall <: ErrorFunction

Optimization goal that searches for a good trade-off between speed and recall (measured
against a gold standard computed with exhaustive search), without requiring a fixed minimum
recall.
"""
struct ParetoRecall <: ErrorFunction end

"""
    ParetoRadius <: ErrorFunction

Optimization goal that searches for a good trade-off between speed and the achieved search
radius, without relying on a computed gold standard.
"""
struct ParetoRadius <: ErrorFunction end

"""
    MaxMatchError(; maxerror=0.1f0, p=1f0, η=1f0, minspread=1f-2) <: ErrorFunction

Optimization goal that favors the fastest configuration among those whose *MatchError* stays
at or below `maxerror`. Unlike [`MinRecall`](@ref) (which compares result and gold *identifiers*
as sets), MatchError compares the *distances* of the returned neighbors against the distances
of the true neighbors at the same rank, so a substitute neighbor tied in distance with the gold
one scores as a perfect match even if its identifier differs (relevant e.g. under `Hamming`,
where many candidates share the same integer distance).

For a query `q`, with `k' = min(k, |gold|)`, gold distances `d*_1 <= ... <= d*_k'` and the
`r` distances actually returned `d_1 <= ... <= d_r` (both ascending):

```
δ_i = max(0, d_i - d*_i) / ρ(q)     for i <= r
δ_i = η                             for i > r   (missing position, penalized)
ρ(q) = d*_k' - min(d*_1, d_1) + minspread + ε
matcherror(q) = mean(δ_i .^ p for i in 1:k')
```

`ρ(q)` is the *spread* of the gold neighborhood (not just its outer radius), so `maxerror`
reads as a fraction of that spread regardless of how dense or sparse this particular query's
neighborhood is — e.g. `maxerror=0.1` means "on average, within 10% of the neighborhood's own
spread beyond where results should be". `0` is a perfect match; the error is unbounded above
(no artificial cap), so a badly-off result keeps registering as worse than a mildly-off one.

`min(d*_1, d_1)` in `ρ(q)` is a deliberate robustness choice: a returned distance below the
gold's own minimum is impossible in theory under a consistent distance function, and in
practice is usually floating-point noise between the exhaustive (gold) pass and the evaluated
index — rather than failing on it (which floating-point noise would trigger often), the range
just absorbs it. A `d_1` far enough below `d*_1` to not be explained by floating-point noise
is instead a sign of a real bug (e.g. a distance function inconsistent with the one used for
the gold standard); this is not currently asserted/validated, only documented here.

`minspread` guards against a genuinely degenerate query: with `k=1`, or whenever the gold
neighborhood's `k'` distances are all tied (routine on real data with near-duplicate/
syndicated items -- e.g. ~2% of queries on a real ccnews slice), the *true* spread
`d*_k' - min(d*_1, d_1)` is exactly `0`, and without a real floor `ρ(q)` collapses to `≈ε`
(machine epsilon) -- dividing by that inflates any ordinary, non-buggy distance mismatch by a
factor of `~10^6-10^7`, so a single such query can swamp a whole batch's mean error. `minspread`
should be picked relative to the typical scale of the distance function in use (e.g. `1f-2` is
reasonable for a `[0, 2]`-ranged cosine-family distance, but Hamming over `nbits` codes wants
something more like `1f0`, one bit); the default is not universally correct, tune it to your
distance.

# Keyword Arguments
- `maxerror`: MatchError threshold (0 is perfect, unbounded above) required to be considered
  as fast as possible.
- `p`: aggregation exponent, `1` for a linear (MAE-like) error, `2` for a quadratic (MSD-like)
  error that suppresses small per-position deviations and amplifies large ones (including
  missing positions, already at `δ_i=η`).
- `η`: penalty assigned to a missing position (the algorithm returned fewer than `k'` items).
- `minspread`: absolute floor added to the gold neighborhood's spread `ρ(q)`, so a fully
  degenerate (zero-spread) query doesn't blow up the aggregate error; see above.

# Examples

```julia
optimize_index!(index, ctx, MaxMatchError(; maxerror=0.1f0, p=2f0))
```
"""
@kwdef struct MaxMatchError <: ErrorFunction
    maxerror::Float32 = 0.1f0
    p::Float32 = 1f0
    η::Float32 = 1f0
    minspread::Float32 = 1f-2
end

"""
    matcherror(golddist::AbstractVector{Float32}, res::AbstractKnnQueue, p::Real, η::Real, minspread::Real=1f-2)::Float64

Per-query MatchError (see [`MaxMatchError`](@ref)): compares the distances actually returned in
`res` against the exact gold distances `golddist` (both compared in ascending rank order),
penalizing missing positions with `η`. `minspread` is the absolute floor added to the gold
neighborhood's spread before normalizing by it, guarding against a degenerate (all-tied) gold
neighborhood -- see [`MaxMatchError`](@ref). Internal function used by
[`create_error_function`](@ref).
"""
function matcherror(golddist::AbstractVector{Float32}, res::AbstractKnnQueue, p::Real, η::Real, minspread::Real=1f-2)::Float64
    kp = length(golddist)
    kp == 0 && return 0.0

    sortitems!(res)
    r = length(res)
    dv = DistView(res)
    dmin = r > 0 ? min(golddist[1], dv[1]) : golddist[1]
    ρ = golddist[kp] - dmin + minspread + eps(Float32)

    s = 0.0
    @inbounds for i in 1:kp
        δ = i <= r ? max(0f0, dv[i] - golddist[i]) / ρ : η
        s += δ^p
    end

    s / kp
end


function setconfig! end

"""
    runconfig(conf, index::AbstractSearchIndex, ctx::AbstractContext, queries::AbstractDatabase, knns::AbstractVector{<:AbstractKnnQueue})

Batch-level counterpart of the single-query `runconfig(conf, index, ctx, q, res)` methods
(e.g. `src/searchgraph/optbs.jl`): runs `conf` against every query in `queries`, in parallel,
mirroring [`searchbatch!`](@ref). Internal function used by [`create_error_function`](@ref).
"""
function runconfig(conf, index::AbstractSearchIndex, ctx::AbstractContext,
                    queries::AbstractDatabase, knns::AbstractVector{<:AbstractKnnQueue})
    m = length(queries)
    minbatch = getminbatch(ctx, m)
    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for i in 1:m
        runconfig(conf, index, bctx, queries[i], reuse!(knns[i]))
    end
    end
    knns
end

"""
    create_error_function(index::AbstractSearchIndex, ctx::AbstractContext, gold, golddists, knns, queries; p=1f0, η=1f0, minspread=1f-2)

Builds and returns a performance-evaluation closure that runs `queries` against `index` under
a candidate configuration and reports its cost (visited nodes), radius, recall (against
`gold`, if given), MatchError (against `golddists`, if given — see [`MaxMatchError`](@ref),
`p`/`η`/`minspread` are its aggregation exponent, missing-position penalty, and degenerate-query
spread floor) and search time. Internal function used by [`optimize_index!`](@ref).
"""
function create_error_function(index::AbstractSearchIndex, ctx::AbstractContext, gold, golddists, knns, queries; p::Float32=1f0, η::Float32=1f0, minspread::Float32=1f-2)
    n = length(index)
    m = length(queries)
    cov = Vector{Float64}(undef, m)
    R = [Set{UInt32}() for _ in knns]

    function lossfun(conf)
        empty!(cov)
        before = copy(ctx.costdists)

        searchtime = @elapsed runconfig(conf, index, ctx, queries, knns)
        searchtime /= m

        for r in knns
            length(r) == maxlength(r) && push!(cov, maximum(r))
        end

        length(cov) <= 3 && throw(InvalidSetupError(conf, "Too few queries fetched k near neighbors"))

        radius = let (rmin, rmax) = extrema(cov)
            while length(cov) < length(knns) # appending maximum radius to increment the mean
                push!(cov, rmax)  ## not so efficient but I hope that this not happens a lot
            end
            (min=rmin, mean=mean(cov), max=rmax)
        end

        recall = if gold !== nothing
            for (i, r) in enumerate(knns)
                empty!(R[i])
                union!(R[i], IdView(r))
            end

            macrorecall(gold, R)
        else
            nothing
        end

        match = if golddists !== nothing
            s = 0.0
            for (i, r) in enumerate(knns)
                s += matcherror(golddists[i], r, p, η, minspread)
            end
            s / m
        else
            nothing
        end

        if recall !== nothing && recall < 0.3
            @warn "OPT low recall> recall: $recall, #objects: $(length(index)), #queries: $(length(queries)), cov: $cov"
            #=for (g, r) in zip(gold, R)
                @show g, r
            end=#

            #=for p in knns
                @show collect(UInt32, I  IdView(p))
            end=#
            #=for p in knns
                @show collect(Float32, DistView(p))
            end=#

            #@show quantile(neighbors_length.(Ref(index.adj), 1:length(index)), 0:0.1:1.0)
            #exit(0)
        end

        visited = distance_stats(ctx, before)
        verbose(ctx) && @inform ctx "error_function> config: $conf, searchtime: $searchtime, recall: $recall, match: $match, length: $(length(index)), radius: $radius, visited: $visited"
        (; visited, radius, recall, match, searchtime, conf)
    end
end


_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize_index!(
        index::AbstractSearchIndex,
        ctx::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        queries=nothing,
        ksearch=10,
        numqueries=64,
        initialpopulation=16,
        maxpopulation=16,
        bsize=4,
        mutbsize=16,
        crossbsize=8,
        maxiters=16,
        params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, maxiters, verbose=verbose(ctx)),
        rng=Random.default_rng()
    )

Tries to configure the `index` to achieve the specified performance (`kind`). The optimization procedure is an stochastic search over the configuration space yielded by `kind` and `queries`.

# Arguments
- `index`: the index to be optimized
- `ctx`: index ctx (caches and general hyperparameters)
- `kind`: The kind of optimization to apply, it can be `ParetoRecall()`, `ParetoRadius()`, `MinRecall(r)` where `r` is the expected recall (0-1, 1 being the best quality but at cost of the search time), or `MaxMatchError(; maxerror)` (a smoother, distance-based alternative to `MinRecall`, see [`MaxMatchError`](@ref))

# Keyword arguments

- `space`: defines the search space
- `queries`: the set of queries to be used to measure performances, a validation set. It can be an `AbstractDatabase` or nothing.
- `ksearch`: the number of neighbors to retrieve for `queries`
- `numqueries`: if `queries===nothing` then a sample of the already indexed database is used, `numqueries` is the size of the sample.
- `rng`: random number generator used to draw the sample of queries when `queries===nothing`.
- `initialpopulation`: the initial sample for the optimization procedure
- `params`: the parameters of the solver, see [`SearchParams` arguments of `SearchModels.jl`](https://github.com/sadit/SearchModels.jl) package for more information.
    Alternatively, you can pass some keywords arguments to `SearchParams`, and use the rest of default values:
    - `initialpopulation=16`: initial sample
    - `maxpopulation=16`: population upper limit
    - `bsize=4`: beam size (top best elements used by select, mutate and crossing operations.)
    - `mutbsize=16`: number of mutated new elements in each iteration
    - `crossbsize=8`: number of new elements from crossing operation.
    - `maxiters=16`: maximum number of iterations.

# Examples

```julia
ctx = SearchGraphContext()
G = SearchGraph(dist, db)
index!(G, ctx)
optimize_index!(G, ctx, MinRecall(0.95))
```
"""
function optimize_index!(
    index::AbstractSearchIndex,
    ctx::AbstractContext,
    kind::ErrorFunction=MinRecall(0.9);
    space::AbstractSolutionSpace=optimization_space(index),
    queries=nothing,
    ksearch=10,
    numqueries=64,
    initialpopulation=16,
    maxpopulation=16,
    bsize=4,
    mutbsize=16,
    crossbsize=8,
    maxiters=16,
    params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, maxiters, verbose=verbose(ctx)),
    rng=Random.default_rng()
)

    db = database(index)
    if queries === nothing
        verbose(ctx) && @inform ctx "using $numqueries random queries from the dataset"
        sample = rand(rng, 1:length(index), numqueries) |> unique
        queries = SubDatabase(db, sample)
    else
        verbose(ctx) && @inform ctx "using $(length(queries)) given as hyperparameter"
    end

    knns_ids = zeros(UInt32, ksearch, length(queries))
    knns_dists = zeros(Float32, ksearch, length(queries))
    knns = [knnqueue(ctx, view(knns_ids, :, i), view(knns_dists, :, i)) for i in 1:length(queries)]
    gold = nothing
    golddists = nothing
    if kind isa ParetoRecall || kind isa MinRecall || kind isa MaxMatchError
        db = @view db[1:length(index)]
        seq = ExhaustiveSearch(distance(index), db)
        searchbatch!(seq, ctx, queries, knns)
        gold = [idset(c) for c in knns]
        if kind isa MaxMatchError
            # `knns` is about to be reused (overwritten) by every candidate evaluated in
            # `create_error_function`, so the gold distances must be copied out now.
            # `sortitems!` mutates `c` in place (a no-op for `KnnSorted`, a real sort for
            # `KnnHeap`) and returns an `IdDistView`, not `c` itself -- read `DistView(c)`
            # from `c` afterwards, not from what `sortitems!` returns.
            golddists = map(knns) do c
                sortitems!(c)
                collect(DistView(c))
            end
        end
    end

    M = Ref(0.0) # max cost
    R = Ref(0.0) # radius
    function inspect_population(space, params, population)
        if M[] == 0.0
            for (c, p) in population
                M[] = max(p.visited.max, M[])
                R[] = max(p.radius.max, R[])
            end
        end
    end

    getperformance = if kind isa MaxMatchError
        create_error_function(index, ctx, gold, golddists, knns, queries; p=kind.p, η=kind.η, minspread=kind.minspread)
    else
        create_error_function(index, ctx, gold, golddists, knns, queries)
    end

    function getcost(p)
        p = last(p)
        cost = p.visited.mean / M[]
        if kind isa ParetoRecall
            cost^2 + (1.0 - p.recall)^2
        elseif kind isa ParetoRadius
            _kfun(cost) + _kfun(p.radius.mean / R[])
        elseif kind isa MinRecall
            #p.recall < kind.minrecall ? 3.0 - 2 * p.recall : cost
            #p.recall < kind.minrecall ? 2f0 - p.recall : cost
            p.recall < kind.minrecall ? 1 + max(kind.minrecall - p.recall, 0) : cost
        elseif kind isa MaxMatchError
            p.match > kind.maxerror ? 1 + max(p.match - kind.maxerror, 0) : cost
        elseif kind isa OptRadius
            r = p.radius.mean / R[]
            round(r / kind.tol, digits=0)
        else
            error("unknown optimization goal $kind")
        end
    end

    function sort_by_best(space, params, population)
        if kind isa OptRadius
            sort!(population, by=getcost)
            sort!(view(population, 1:params.bsize), by=p -> p.second.visited.mean)
        else
            sort!(population, by=getcost)
        end

        population
    end

    function convergence(curr, prev)
        abs(getcost(prev) - getcost(curr)) <= 1e-3
    end

    bestlist = search_models(getperformance, space, initialpopulation, params; inspect_population, sort_by_best, convergence, parallel=:none)

    if length(bestlist) == 0
        verbose(ctx) && @inform ctx "== WARN optimization failure; unable to find usable configurations"
    else
        config, perf = bestlist[1]
        # @assert perf.recall > 0
        verbose(ctx) && @inform ctx "== finished opt. $(typeof(index)): search-params: $(params), opt-config: $config, perf: $perf, kind=$(kind), length=$(length(index))"
        setconfig!(config, index, perf)
    end

    bestlist
end

