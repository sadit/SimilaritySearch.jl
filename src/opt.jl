# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize_index!, MinRecall, OptRadius, ParetoRecall, ParetoRadius

abstract type ErrorFunction end
@kwdef struct MinRecall <: ErrorFunction
    minrecall::Float32 = 0.9f0
end

@kwdef struct OptRadius <: ErrorFunction
    tol::Float32 = 0.1
end

struct ParetoRecall <: ErrorFunction end
struct ParetoRadius <: ErrorFunction end


function setconfig! end

function create_error_function(index::AbstractSearchIndex, ctx::AbstractContext, gold, knns, queries)
    n = length(index)
    m = length(queries)
    cost = zeros(Int, m)
    cov = Vector{Float64}(undef, m)
    R = [Set{Int32}() for _ in knns]

    function lossfun(conf)
        empty!(cov)

        searchtime = @elapsed begin
            minbatch = getminbatch(ctx, m)
            Threads.@threads :static for j in 1:minbatch:m
                for i in j:min(m, j + minbatch - 1)
                    knns[i] = r = runconfig(conf, index, ctx, queries[i], reuse!(knns[i]))
                    cost[i] = distance_evaluations(r)
                end
            end
        end

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

        if recall < 0.3
            @warn "OPT low recal> recall: $recall, #objects: $(length(index)), #queries: $(length(queries))"
            @show cov
            #=for (g, r) in zip(gold, R)
                @show g, r
            end=#

            #=for p in knns
                @show collect(Int32, I  IdView(p))
            end=#
            #=for p in knns
                @show collect(Float32, DistView(p))
            end=#

            #@show quantile(neighbors_length.(Ref(index.adj), 1:length(index)), 0:0.1:1.0)
            #exit(0)
        end

        visited = (min=minimum(cost), mean=mean(cost), max=maximum(cost))
        verbose(ctx) && println(stderr, "error_function> config: $conf, searchtime: $searchtime, recall: $recall, length: $(length(index)), radius: $radius, visited: $visited")
        (; visited, radius, recall, searchtime, conf)
    end
end


_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize_index!(
        index::AbstractSearchIndex,
        ctx::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        ctx=GenericContext(ctx),
        queries=nothing,
        ksearch=10,
        numqueries=64,
        initialpopulation=16,
        maxpopulation=16,
        bsize=4,
        mutbsize=16,
        crossbsize=8,
        tol=-1.0,
        maxiters=16,
        params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, tol, maxiters, verbose=verbose(ctx))
    )

Tries to configure the `index` to achieve the specified performance (`kind`). The optimization procedure is an stochastic search over the configuration space yielded by `kind` and `queries`.

# Arguments
- `index`: the index to be optimized
- `ctx`: index ctx (caches and general hyperparameters)
- `kind`: The kind of optimization to apply, it can be `ParetoRecall()`, `ParetoRadius()` or `MinRecall(r)` where `r` is the expected recall (0-1, 1 being the best quality but at cost of the search time)

# Keyword arguments

- `space`: defines the search space
- `queries`: the set of queries to be used to measure performances, a validation set. It can be an `AbstractDatabase` or nothing.
- `queries_ksearch`: the number of neighbors to retrieve for `queries`
- `queries_size`: if `queries===nothing` then a sample of the already indexed database is used, `queries_size` is the size of the sample.
- `initialpopulation`: the initial sample for the optimization procedure
- `params`: the parameters of the solver, see [`SearchParams` arguments of `SearchModels.jl`](https://github.com/sadit/SearchModels.jl) package for more information.
    Alternatively, you can pass some keywords arguments to `SearchParams`, and use the rest of default values:
    - `initialpopulation=16`: initial sample
    - `maxpopulation=16`: population upper limit
    - `bsize=4`: beam size (top best elements used by select, mutate and crossing operations.)
    - `mutbsize=16`: number of mutated new elements in each iteration
    - `crossbsize=8`: number of new elements from crossing operation.
    - `maxiters=16`: maximum number of iterations.
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
        verbose(ctx) && @info "using $numqueries random queries from the dataset"
        sample = rand(rng, 1:length(index), numqueries) |> unique
        queries = SubDatabase(db, sample)
    else
        verbose(ctx) && @info "using $(length(queries)) given as hyperparameter"
    end

    knnsmatrix = zeros(IdWeight, ksearch, length(queries))
    knns = [knnqueue(ctx, c) for c in eachcol(knnsmatrix)]
    gold = nothing
    if kind isa ParetoRecall || kind isa MinRecall
        db = @view db[1:length(index)]
        seq = ExhaustiveSearch(distance(index), db)
        knns = searchbatch!(seq, ctx, queries, knns)
        gold = [idset(c) for c in knns]
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

    getperformance = create_error_function(index, ctx, gold, knns, queries)

    function getcost(p)
        p = last(p)
        cost = p.visited.mean / M[]
        if kind isa ParetoRecall
            cost^2 + (1.0 - p.recall)^2
        elseif kind isa ParetoRadius
            _kfun(cost) + _kfun(p.radius.mean / R[])
        elseif kind isa MinRecall
            p.recall < kind.minrecall ? 3.0 - 2 * p.recall : cost
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
        verbose(ctx) && println(stderr, "== WARN optimization failure; unable to find usable configurations")
    else
        config, perf = bestlist[1]
        # @assert perf.recall > 0
        verbose(ctx) && println(stderr, "== finished opt. $(typeof(index)): search-params: $(params), opt-config: $config, perf: $perf, kind=$(kind), length=$(length(index))")
        setconfig!(config, index, perf)
    end

    bestlist
end

