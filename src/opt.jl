# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize!, MinRecall, ParetoRecall, ParetoRadius

abstract type ErrorFunction end
@with_kw struct MinRecall <: ErrorFunction
    minrecall = 0.9
end

struct ParetoRecall <: ErrorFunction end
struct ParetoRadius <: ErrorFunction end

function runconfig end
function setconfig! end

function create_error_function(index::AbstractSearchIndex, gold, knns::KnnResultSet, queries, ksearch, verbose)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    numqueries = size(knns, 2)
    covradius = Vector{Float64}(undef, numqueries)
    pools = getpools(index)

    function lossfun(conf)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vacc .= 0.0
        
        searchtime = @elapsed begin
            Threads.@threads for i in 1:length(queries)
                _, c = runconfig(conf, index, queries[i], reuse!(knns, i), pools)
                ti = Threads.threadid()
                vmin[ti] = min(c, vmin[ti])
                vmax[ti] = max(c, vmax[ti])
                vacc[ti] += c
            end
        end

        for (i, l) in enumerate(knns.len)
            covradius[i] = l == 0 ? typemax(Float32) : knns.dist[l, i]
        end

        rmin, rmax = extrema(covradius)
        ravg = mean(covradius)

        recall = if gold !== nothing
            macrorecall(gold, knns.id)
        else
            nothing
        end

        verbose && println(stderr, "error_function> config: $conf, searchtime: $searchtime, recall: $recall, length: $(length(index))")
        (;
            visited=(minimum(vmin), sum(vacc)/numqueries, maximum(vmax)),
            radius=(rmin, ravg, rmax),
            recall=recall,
            searchtime=searchtime/numqueries
        )
    end
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize!(
        index::AbstractSearchIndex,
        kind::ErrorFunction=ParetoRecall(),
        space::AbstractSolutionSpace=optimization_space(index);
        queries=nothing,
        ksearch=10,
        numqueries=64,
        initialpopulation=8,
        verbose=false,
        params=SearchParams(; maxpopulation=8, bsize=4, mutbsize=8, crossbsize=2, tol=-1.0, maxiters=8, verbose)
    )

Tries to configure the `index` to achieve the specified performance (`kind`). The optimization procedure is an stochastic search over the configuration space yielded by `kind` and `queries`.

# Arguments
- `index`: the index to be optimized
- `kind`: The kind of optimization to apply, it can be `ParetoRecall()`, `ParetoRadius()` or `MinRecall(r)` where `r` is the expected recall (0-1, 1 being the best quality but at cost of the search time)

# Keyword arguments

- `queries`: the set of queries to be used to measure performances, a validation set. It can be an `AbstractDatabase` or nothing.
- `queries_ksearch`: the number of neighbors to retrieve for `queries`
- `queries_size`: if `queries===nothing` then a sample of the already indexed database is used, `queries_size` is the size of the sample.
- `initialpopulation`: the initial sample for the optimization procedure
- `space`: defines the search space
- `params`: the parameters of the solver, see [`search_models` function from `SearchModels.jl`](https://github.com/sadit/SearchModels.jl) package for more information.
- `verbose`: controls if the procedure is verbose or not
"""
function optimize!(
            index::AbstractSearchIndex,
            kind::ErrorFunction=ParetoRecall(),
            space::AbstractSolutionSpace=optimization_space(index);
            queries=nothing,
            ksearch=10,
            numqueries=64,
            initialpopulation=16,
            parallel=true,
            verbose=false,
            params=SearchParams(; maxpopulation=16, bsize=4, mutbsize=16, crossbsize=8, tol=-1.0, maxiters=16, verbose)
    )

    if queries === nothing
        sample = rand(1:length(index), numqueries) |> unique
        queries = SubDatabase(index.db, sample)
    end

    knns = KnnResultSet(ksearch, length(queries))
    gold = nothing
    if kind isa ParetoRecall || kind isa MinRecall
        db = @view index.db[1:length(index)]
        seq = ExhaustiveSearch(index.dist, db)
        searchbatch(seq, queries, knns; parallel)
        gold = copy(knns.id)
    end

    M = Ref(0.0)
    R = Ref(0.0)
    function inspect_pop(space, params, population)
        if M[] == 0.0
            for (c, p) in population
                M[] = max(p.visited[end], M[])
                R[] = max(p.radius[end], R[])
            end
        end

    end

    errorfun = create_error_function(index, gold, knns, queries, ksearch, verbose)

    function geterr(p)
        cost = p.visited[2] / M[]
        if kind isa ParetoRecall 
            cost^2 + (1.0 - p.recall)^2
        elseif kind isa MinRecall
            p.recall < kind.minrecall ? 3.0 - 2 * p.recall : cost
        else
            _kfun(cost) + _kfun(p.radius[2] / R[])
        end
    end
    
    bestlist = search_models(
        errorfun,
        space, 
        initialpopulation,
        params;
        inspect_population=inspect_pop,
        geterr=geterr)
    
    config, perf = bestlist[1]
    verbose && println(stderr, "== finished opt. $(typeof(index)): search-params: $(params), opt-config: $config, perf: $perf, kind=$(kind), length=$(length(index))")
    setconfig!(config, index, perf)
        bestlist
end