# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize_index!, MinRecall, ParetoRecall, ParetoRadius

abstract type ErrorFunction end
@with_kw struct MinRecall <: ErrorFunction
    minrecall = 0.9
end

struct ParetoRecall <: ErrorFunction end
struct ParetoRadius <: ErrorFunction end

function runconfig0(conf, index::AbstractSearchIndex, ctx::AbstractContext, queries::AbstractDatabase, i::Integer, res::KnnResult)
    runconfig(conf, index, ctx, queries[i], res)
end

function setconfig! end

function create_error_function(index::AbstractSearchIndex, context::AbstractContext, gold, knnlist::Vector{KnnResult}, queries, ksearch, verbose)
    n = length(index)
    m = length(queries)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    cov = Vector{Float64}(undef, m)
    R = [Set{Int32}() for _ in knnlist]

    function lossfun(conf)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vacc .= 0.0
        
        searchtime = @elapsed begin
            @batch minbatch=getminbatch(0, m) per=thread for i in 1:m
                r_ = runconfig0(conf, index, context, queries, i, reuse!(knnlist[i], ksearch))
                ti = Threads.threadid()
                vmin[ti] = min(r_.cost, vmin[ti])
                vmax[ti] = max(r_.cost, vmax[ti])
                vacc[ti] += r_.cost
            end
        end

        for i in eachindex(knnlist)
            res = knnlist[i]
            cov[i] = covradius(res)
        end

        rmin, rmax = extrema(cov)
        ravg = mean(cov)

        recall = if gold !== nothing
            for (i, res) in enumerate(knnlist)
                empty!(R[i])
                union!(R[i], IdView(res))
            end

            macrorecall(gold, R)
        else
            nothing
        end

        verbose && println(stderr, "error_function> config: $conf, searchtime: $searchtime, recall: $recall, length: $(length(index))")
        (;
            visited=(minimum(vmin), sum(vacc)/m, maximum(vmax)),
            radius=(rmin, ravg, rmax),
            recall=recall,
            searchtime=searchtime/m
        )
    end
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize_index!(
        index::AbstractSearchIndex,
        context::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        context_exhaustive_search=GenericContext(context),
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
        verbose=false,
        params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, tol, maxiters, verbose)
    )

Tries to configure the `index` to achieve the specified performance (`kind`). The optimization procedure is an stochastic search over the configuration space yielded by `kind` and `queries`.

# Arguments
- `index`: the index to be optimized
- `context`: index context
- `kind`: The kind of optimization to apply, it can be `ParetoRecall()`, `ParetoRadius()` or `MinRecall(r)` where `r` is the expected recall (0-1, 1 being the best quality but at cost of the search time)

# Keyword arguments

- `space`: defines the search space
- `queries`: the set of queries to be used to measure performances, a validation set. It can be an `AbstractDatabase` or nothing.
- `queries_ksearch`: the number of neighbors to retrieve for `queries`
- `queries_size`: if `queries===nothing` then a sample of the already indexed database is used, `queries_size` is the size of the sample.
- `initialpopulation`: the initial sample for the optimization procedure
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `params`: the parameters of the solver, see [`SearchParams` arguments of `SearchModels.jl`](https://github.com/sadit/SearchModels.jl) package for more information.
    Alternatively, you can pass some keywords arguments to `SearchParams`, and use the rest of default values:
    - `initialpopulation=16`: initial sample
    - `minbatch=0`: minimum batch size (`Polyester` multithreading, `0` chooses the size based on the input)
    - `maxpopulation=16`: population upper limit
    - `bsize=4`: beam size (top best elements used by select, mutate and crossing operations.)
    - `mutbsize=16`: number of mutated new elements in each iteration
    - `crossbsize=8`: number of new elements from crossing operation.
    - `tol=-1.0`: tolearance change between iterations (negative values means disables stopping by converguence)
    - `maxiters=16`: maximum number of iterations.
    - `verbose`: controls if the procedure is verbose or not
"""
function optimize_index!(
        index::AbstractSearchIndex,
        context::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        context_exhaustive_search=GenericContext(context),
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
        verbose=false,
        params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, tol, maxiters, verbose)
    )

    db = database(index)
    if queries === nothing
        sample = rand(1:length(index), numqueries) |> unique
        queries = SubDatabase(db, sample)
    end

    knnlist = [KnnResult(ksearch) for _ in eachindex(queries)]
    gold = nothing
    if kind isa ParetoRecall || kind isa MinRecall
        db = @view db[1:length(index)]
        seq = ExhaustiveSearch(distance(index), db)
        searchbatch(seq, context_exhaustive_search, queries, knnlist)
        gold = [Set(item.id for item in res) for res in knnlist]
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

    errorfun = create_error_function(index, context, gold, knnlist, queries, ksearch, verbose)

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

