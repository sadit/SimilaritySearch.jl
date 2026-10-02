# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
using Statistics: median, std
import SearchModels: combine, mutate
export OptimizeParameters, optimize_index!, MinRecall, MaxMatchError

"""
    abstract type ErrorFunction end

Abstract type for the optimization goals (`kind` argument) accepted by [`optimize_index!`](@ref).
It determines how candidate hyperparameter configurations are scored/compared while
autotuning the index. Concrete subtypes are [`MinRecall`](@ref) and [`MaxMatchError`](@ref).

Three goals were removed in 1.6. `ParetoRecall`/`ParetoRadius` were a weighted sum of squares
normalized by the initial population's maximum cost, not a Pareto front, and the trade-off
they picked at construction did not carry over to the search; a bi-objective goal, when it
returns, will be a smooth, explicitly weighted combination. `OptRadius` targeted a covering
radius within a tolerance, which asked for a look at the distances beforehand to be set at
all; `MaxMatchError` is the same idea with the scale read off each query's own neighborhood.
"""
abstract type ErrorFunction end

"""
    MinRecall(minrecall=0.9f0; tradeoff=1.5, width=nothing, transition=(-1, 1)) <: ErrorFunction

Optimization goal: the cheapest configuration whose recall (against a gold standard computed
with exhaustive search) reaches `minrecall`, and, below it, the one whose saving pays for its
shortfall. The value minimized is [`goalvalue`](@ref),

    log(visits) + tradeoffrate · hinge(minrecall − recall)

a smooth hinge on the target rather than a threshold. Far above `minrecall` only the cost
counts; far below, every unit of recall missing costs `tradeoffrate = log(tradeoff) / 0.01`
nats of cost, and the cost still counts; across a transition zone on the scale of the
measurement's own noise, `width`, the hinge is quadratic, so a configuration a hair short of
the target is priced at that hair instead of discarded, and the ranking does not flip with
the noise of the recall estimate. (The threshold it replaces ranked anything below the
target behind anything above it, whatever the costs, and with 64 tuning queries the
estimate's standard error is about 0.04, so the test was a coin toss for every configuration
within that of the target.) The hinge has finite support: it is exactly zero before the
transition zone, so the goal never pushes the recall further above the target than that
zone reaches -- a `softplus` hinge was measured to overshoot the target by two to three
widths, growing with `tradeoff`, because its tail never reaches zero.

# Arguments
- `minrecall`: the target recall (0-1).
- `tradeoff`: the cost factor accepted per 1% of recall near the target: `1.5` reads "up to 50%
  more visits for each 1% of recall". Must be finite and `> 1`. Above the slope of the
  cost-against-recall front -- 10-20 nats per unit of recall on typical graphs, a cost that
  doubles between recall 0.90 and 0.95 -- the minimum is the one a hard constraint would pick;
  `1.5` is 40. With the default zone the result barely moves between `1.2` and `3` (measured
  on SISAP 2025 `ccnews`: recall 0.900-0.913 for a target of 0.9).
- `width`: the unit of the transition zone, in recall units. `nothing` (the default) derives
  it from the data in [`optimize_index!`](@ref): the standard error of the macro recall over
  the tuning queries, as the median over the initial population of
  `std(per-query recall) / sqrt(numqueries)`, so the zone spans exactly what the measurement
  cannot tell apart. A small explicit `width` shrinks the zone to a hard threshold.
- `transition`: the transition zone `(lo, hi)`, as multipliers of `width`, in shortfall
  units `s = minrecall − recall` (positive below the target). The hinge charges nothing for
  `s ≤ lo · width`, a quadratic across the zone, and the shortfall itself, up to a constant,
  from `hi · width` on; it is continuous with a continuous derivative for any `lo ≤ hi`, and
  `lo == hi` is a hard threshold at `lo · width`. Measured on SISAP 2025 `ccnews`, a target of
  0.9, 64 or 256 tuning queries, `width` derived:

  | `transition` | zone, in widths | at the target | tuned recall lands |
  |---|---|---|---|
  | `(-1, 1)`, the default | one width on each side | `width / 4` | within a width above the target: 0.900-0.913 |
  | `(0, 2)` | starts at the target | `0` | like a hard threshold, at or a little under the target with few queries: 0.877-0.901 |
  | `(-2, 0)` | ends at the target | `width` (already linear) | one to two widths above: for a target that is a floor to hold on unseen queries |

  Any other pair works the same way: `(-0.5, 0.5)` trusts the measurement more than its own
  standard error, `(-1, 3)` is lenient below the target and strict above it.

# Arguments
- `minrecall`: the target recall (0-1).
- `tradeoff`: the cost factor accepted per 1% of recall near the target: `1.5` reads "up to 50%
  more visits for each 1% of recall". Must be finite and `> 1`. Above the slope of the
  cost-against-recall front -- 10-20 nats per unit of recall on typical graphs, a cost that
  doubles between recall 0.90 and 0.95 -- the minimum is the one a hard constraint would pick;
  `1.5` is 40.
- `width`: the half-width of the smooth transition, in recall units. `nothing` (the default)
  derives it from the data in [`optimize_index!`](@ref): the standard error of the macro recall
  over the tuning queries, as the median over the initial population of
  `std(per-query recall) / sqrt(numqueries)`, so the objective is flat exactly where the
  measurement cannot tell configurations apart. A small explicit `width` recovers a hard
  threshold.

# Examples

```julia
optimize_index!(index, ctx, MinRecall(0.95))
optimize_index!(index, ctx, MinRecall(0.95; tradeoff=3.0))        # a shortfall is cheaper to accept
optimize_index!(index, ctx, MinRecall(0.95; transition=(-2, 0)))  # 0.95 is a floor, land above it
```
"""
@kwdef struct MinRecall <: ErrorFunction
    minrecall::Float32 = 0.9f0
    tradeoff::Float64 = 1.5
    width::Union{Nothing,Float32} = nothing
    transition::Tuple{Float32,Float32} = (-1f0, 1f0)

    function MinRecall(minrecall, tradeoff, width, transition)
        _checkgoal("MinRecall", tradeoff, width, transition)
        new(Float32(minrecall), Float64(tradeoff), width === nothing ? nothing : Float32(width), _transition(transition))
    end
end

MinRecall(minrecall::Real; tradeoff::Real=1.5, width=nothing, transition=(-1, 1)) = MinRecall(minrecall, tradeoff, width, transition)

_transition(t) = (Float32(t[1]), Float32(t[2]))

function _checkgoal(name, tradeoff, width, transition)
    tradeoff > 1 && isfinite(tradeoff) || throw(ArgumentError("$name: tradeoff=$tradeoff must be a finite number above 1"))
    width === nothing || width > 0 || throw(ArgumentError("$name: width=$width must be positive"))
    length(transition) == 2 && all(isfinite, transition) && transition[1] <= transition[2] ||
        throw(ArgumentError("$name: transition=$transition must be a pair (lo, hi) of finite width multipliers with lo <= hi"))
    nothing
end

"""
    MaxMatchError(; maxerror=0.1f0, p=1f0, η=1f0, minspread=1f-2, tradeoff=1.5, width=nothing, transition=(-1, 1)) <: ErrorFunction

Optimization goal: the cheapest configuration whose *MatchError* stays at or below
`maxerror`, and, above it, the one whose saving pays for the excess -- the same smooth hinge
as [`MinRecall`](@ref), [`goalvalue`](@ref) with `matcherror − maxerror` as the shortfall and
`width` in match-error units. Unlike [`MinRecall`](@ref) (which compares result and gold *identifiers*
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
- `tradeoff`: the cost factor accepted per 0.01 of match error near `maxerror`; must be finite
  and `> 1` (see [`MinRecall`](@ref)).
- `width`: the unit of the hinge's transition zone, in match-error units; `nothing` derives it
  in `optimize_index!` as the standard error of the macro match error over the tuning queries
  (the median over the initial population of `std(per-query matcherror) / sqrt(numqueries)`).
- `transition`: the zone `(lo, hi)` as multipliers of `width`, placed on `maxerror` (see
  [`MinRecall`](@ref); here the shortfall is `matcherror − maxerror`).

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
    tradeoff::Float64 = 1.5
    width::Union{Nothing,Float32} = nothing
    transition::Tuple{Float32,Float32} = (-1f0, 1f0)

    function MaxMatchError(maxerror, p, η, minspread, tradeoff, width, transition)
        _checkgoal("MaxMatchError", tradeoff, width, transition)
        new(Float32(maxerror), Float32(p), Float32(η), Float32(minspread), Float64(tradeoff), width === nothing ? nothing : Float32(width), _transition(transition))
    end
end

"""
    goalvalue(kind::MinRecall, visits::Real, recall::Real; width=kind.width) -> Float64
    goalvalue(kind::MaxMatchError, visits::Real, matcherror::Real; width=kind.width) -> Float64

The number [`optimize_index!`](@ref) minimizes for a configuration that visited `visits`
objects per query and measured the given quality:

    log(visits) + tradeoffrate · hinge(shortfall)

with `tradeoffrate = log(kind.tradeoff) / 0.01` and `shortfall` how far the quality falls
short of the goal's target, `minrecall − recall` or `matcherror − maxerror`. The cost is in
nats, so a difference of `log(2)` is "twice the visits" at any scale and no normalization is
needed. The hinge has finite support: with the transition zone `a = lo · width` to
`b = hi · width` from `kind.transition`, it is exactly `0` before the zone, a quadratic
across it, and the shortfall up to a constant beyond it, continuous with a continuous
derivative:

    hinge(s) = 0                        for s ≤ a
             = (s − a)² / (2 (b − a))   for a < s < b
             = s − (a + b) / 2          for s ≥ b

For the default `(-1, 1)` that is `(s + width)² / (4 width)` across `[−width, width]` and
`s` beyond.

`width` must be resolved: given to the goal, passed here, or derived by `optimize_index!`
from the initial population before the first ranking.
"""
goalvalue(kind::MinRecall, visits::Real, recall::Real; width=kind.width) =
    _goalvalue(visits, kind.minrecall - recall, kind.tradeoff, width, kind.transition)
goalvalue(kind::MaxMatchError, visits::Real, matcherror::Real; width=kind.width) =
    _goalvalue(visits, matcherror - kind.maxerror, kind.tradeoff, width, kind.transition)

function _goalvalue(visits::Real, shortfall::Real, tradeoff::Real, width, transition)::Float64
    width === nothing && throw(ArgumentError("goalvalue: the goal's width is unresolved; give it to the goal or let optimize_index! derive it"))
    rate = log(tradeoff) / 0.01
    log(max(Float64(visits), 1.0)) + rate * _hinge(Float64(shortfall), Float64(width), transition)
end

"""
The finite-support hinge over the zone `[lo · width, hi · width]`: `0` before it, a quadratic
across it, the shortfall up to a constant after it, with a continuous derivative. `lo == hi`
is a plain threshold at `lo · width`.
"""
function _hinge(s::Float64, width::Float64, transition)::Float64
    a = Float64(transition[1]) * width
    b = Float64(transition[2]) * width
    s <= a && return 0.0
    s >= b && return s - (a + b) / 2
    (s - a)^2 / (2 * (b - a))
end

"""
    matcherror(golddist, res, err::MaxMatchError) -> Float64
    macromatcherror(golddists, reslist, err::MaxMatchError) -> Float64

The per-query and the macro [`matcherror`](@ref) with the parameters `p`, `η` and `minspread`
taken from `err`, so a score can be computed outside the optimizer exactly as
[`optimize_index!`](@ref) computes it: `bootstrapscore((g, r) -> matcherror(g, r, err), golddists, reslist)`.
"""
matcherror(golddist, res, err::MaxMatchError) = matcherror(golddist, res, err.p, err.η, err.minspread)
macromatcherror(golddists, reslist, err::MaxMatchError) = macromatcherror(golddists, reslist, err.p, err.η, err.minspread)

function setconfig! end

"""
    qid(index::AbstractSearchIndex, queries::AbstractDatabase, i::Integer) -> UInt32

The identifier of the `i`-th query *inside `index`*, or `0` when the query does not live
there. It is what tells an optimization run whether it is working with **internal** queries
(objects taken from the index's own database) or **external** ones, which behave differently
enough that the distinction has to be explicit:

An internal query is a vertex of the graph. The descent can stand on it at distance 0 and
read its whole adjacency in one expansion -- and that adjacency is, by construction,
approximately the answer. No external query is ever handed its result that way, so tuning
against internal queries without accounting for it produces parameters sized for a problem
nobody will pose: measured on two SISAP 2025 benchmarks, `bsize` and `Δ` came out at the
cheap end of their ranges and recall@10 against real queries fell from 0.90 to 0.69.

The identity check on `parent` is what makes this exact: a `SubDatabase` over *this* index's
database carries the ids in `map`, and a view over anything else is external, as is any other
container. It also means a caller who wants to tune with chosen objects of the database --
the least connected ones, say -- only has to pass `SubDatabase(database(index), ids)`.
"""
function qid end

@inline qid(::AbstractSearchIndex, ::AbstractDatabase, ::Integer) = zero(UInt32)
@inline qid(index::AbstractSearchIndex, q::SubDatabase, i::Integer) =
    q.parent === database(index) ? UInt32(@inbounds q.map[i]) : zero(UInt32)

"""
    runconfig(conf, index::AbstractSearchIndex, ctx::AbstractContext, q, qID::Integer, res::AbstractKnnQueue)

Fallback for index types that do not act on the internal/external distinction: the `qID` is
dropped and the plain single-query method runs. `SearchGraph` overrides it (see
`src/searchgraph/optbs.jl`) to keep an internal query from being its own route.
"""
runconfig(conf, index::AbstractSearchIndex, ctx::AbstractContext, q, ::Integer, res::AbstractKnnQueue) =
    runconfig(conf, index, ctx, q, res)

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
        bctx = beginbatch(ctx, @batchid())
    @LOOP for i in 1:m
        runconfig(conf, index, bctx, queries[i], qid(index, queries, i), reuse!(knns[i]))
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

        # the macro scores and their standard errors over the tuning queries; the goals' smooth
        # hinge takes its width from the latter when the goal did not fix one
        recall, recallstd = if gold !== nothing
            for (i, r) in enumerate(knns)
                empty!(R[i])
                union!(R[i], IdView(r))
            end

            pq = perqueryscores(recallscore, gold, R)
            mean(pq), std(pq) / sqrt(m)
        else
            nothing, nothing
        end

        match, matchstd = if golddists !== nothing
            pq = perqueryscores((g, r) -> matcherror(g, r, p, η, minspread), golddists, knns)
            mean(pq), std(pq) / sqrt(m)
        else
            nothing, nothing
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
        (; visited, radius, recall, recallstd, match, matchstd, searchtime, conf)
    end
end


"""
    optimize_index!(
        index::AbstractSearchIndex,
        ctx::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        queries=nothing,
        ksearch=10,
        radius=nothing,
        kmin=8,
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
- `kind`: the goal, [`MinRecall`](@ref)`(r)` with `r` the target recall (0-1) or [`MaxMatchError`](@ref)`(; maxerror)`, its distance-based counterpart; both minimize [`goalvalue`](@ref), the log cost plus a smooth hinge on the target

# Keyword arguments

- `space`: defines the search space
- `queries`: the set of queries to be used to measure performances, a validation set. It can be an `AbstractDatabase` or nothing.
- `ksearch`: the number of neighbors to retrieve for `queries` (k-NN workloads only; ignored when `radius` is given)
- `radius`: tune for radius-bounded (epsilon-ball) queries of this radius instead of k-NN queries.
  The gold standard becomes each query's true ball -- of whatever size, empty included -- and
  candidates are scored with [`matcherror`](@ref) over it, so this requires `kind::MaxMatchError`;
  the recall-based goals raise an `ArgumentError`, since `macrorecall` divides by the gold ball's
  size and a small radius routinely produces empty balls. What gets tuned is an ordinary
  `BeamSearch`, so the result also governs later k-NN searches on the index.
- `kmin`: navigation reserve used while tuning (see [`BallKnn`](@ref)); pass the value the radius
  searches themselves will use, since a configuration is only tuned relative to it
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
    radius=nothing,
    kmin::Int=8,
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

    gold = nothing
    golddists = nothing

    knns = if radius === nothing
        knns_ids = zeros(UInt32, ksearch, length(queries))
        knns_dists = zeros(Float32, ksearch, length(queries))
        [knnqueue(ctx, view(knns_ids, :, i), view(knns_dists, :, i)) for i in 1:length(queries)]
    else
        # Radius workload: tune against balls instead of k nearest neighbors. Only MaxMatchError
        # transfers -- see `matcherror(::Any, ::BallKnn, ...)` for why the recall-based goals
        # cannot. The containers are `BallKnn` rather than `RadiusSorted` for two reasons: the
        # search needs the navigation reserve to reach the ball at all (#67), and `lossfun` records
        # a covering radius only where `length(r) == maxlength(r)`, which a container of unbounded
        # capacity never satisfies -- it would reject every configuration with InvalidSetupError.
        kind isa MaxMatchError || throw(ArgumentError("optimize_index!: radius=$radius requires kind::MaxMatchError, got $(typeof(kind)); the recall-based goals score with macrorecall, which divides by the gold ball's size, and a small radius routinely yields empty balls"))
        [BallKnn(radius, kmin) for _ in 1:length(queries)]
    end

    if kind isa MinRecall || kind isa MaxMatchError
        db = @view db[1:length(index)]
        seq = ExhaustiveSearch(distance(index), db)
        searchbatch!(seq, ctx, queries, knns)
        # `knns` is about to be reused (overwritten) by every candidate evaluated in
        # `create_error_function`, so the gold must be copied out now. `sortitems!` mutates `c`
        # in place (a no-op for `KnnSorted`, a real sort for `KnnHeap`) and returns an
        # `IdDistView`, not `c` itself -- read `DistView(c)` from `c` afterwards, not from what
        # `sortitems!` returns.
        if radius === nothing
            # An internal query is in its own gold, at distance 0. It is also masked out of the
            # search that is being scored (see `runconfig` for `SearchGraph`), so leaving it in
            # the gold would cap recall at (k-1)/k -- 0.9 for the default k, which would put
            # the 0.97 construction target out of reach. Both sides drop it, and `recallscore`
            # normalizes by `length(gold)`, so nothing else has to change. `qid` is 0 for
            # external queries, which match no identifier and lose nothing.
            gold = map(enumerate(knns)) do (i, c)
                g = idset(c)
                delete!(g, qid(index, queries, i))
                g
            end

            if kind isa MaxMatchError
                golddists = map(enumerate(knns)) do (i, c)
                    id = qid(index, queries, i)
                    Float32[p.dist for p in sortitems!(c) if p.id != id]
                end
            end
        else
            # the exhaustive pass filled every BallKnn with the *true* ball (plus a reserve, which
            # is not part of the gold); `gold` stays nothing, as recall is not computed here
            golddists = [Float32[p.dist for p in ballview(c)] for c in knns]
        end
    end

    # the goal's hinge width: given, or the standard error of the quality measurement over the
    # tuning queries, read off the initial population once (its median, so one odd
    # configuration does not set it) -- the objective is flat where the measurement is blind
    width = Ref{Union{Nothing,Float64}}(kind.width === nothing ? nothing : Float64(kind.width))
    function inspect_population(space, params, population)
        if width[] === nothing
            stds = Float64[]
            for (c, perf) in population
                s = kind isa MinRecall ? perf.recallstd : perf.matchstd
                s === nothing || isnan(s) || push!(stds, s)
            end
            width[] = isempty(stds) ? 1e-3 : max(median(stds), 1e-4)
            verbose(ctx) && @inform ctx "== goal width resolved to $(width[]) (standard error of the quality over $(length(queries)) tuning queries)"
        end
    end

    getperformance = if kind isa MaxMatchError
        create_error_function(index, ctx, gold, golddists, knns, queries; p=kind.p, η=kind.η, minspread=kind.minspread)
    else
        create_error_function(index, ctx, gold, golddists, knns, queries)
    end

    function getcost(p)
        perf = last(p)
        quality = kind isa MinRecall ? perf.recall : perf.match
        goalvalue(kind, perf.visited.mean, quality; width=width[])
    end

    function sort_by_best(space, params, population)
        sort!(population, by=getcost)
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

