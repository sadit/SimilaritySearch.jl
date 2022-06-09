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

function create_error_function(index::AbstractSearchContext, gold, knnlist::Vector{KnnResult}, queries, ksearch, verbose)
    n = length(index)
    m = length(queries)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    covradius = Vector{Float64}(undef, length(knnlist))
    pools = getpools(index)
    R = [Set{Int32}() for _ in knnlist]

    function lossfun(conf)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vacc .= 0.0
        
        searchtime = @elapsed begin
            @batch minbatch=getminibatch(0, m) per=thread for i in 1:m
                knnlist[i] = reuse!(knnlist[i], ksearch)
                _, v_ = runconfig(conf, index, queries[i], knnlist[i], pools)
                ti = Threads.threadid()
                vmin[ti] = min(v_, vmin[ti])
                vmax[ti] = max(v_, vmax[ti])
                vacc[ti] += v_
            end
        end

        for i in eachindex(knnlist)
            res = knnlist[i]
            covradius[i] = length(res) == 0 ? typemax(Float32) : maximum(res)
        end

        rmin, rmax = extrema(covradius)
        ravg = mean(covradius)

        recall = if gold !== nothing
            for (i, res) in enumerate(knnlist)
                empty!(R[i])
                union!(R[i], res.id)
            end

            macrorecall(gold, R)
        else
            nothing
        end

        verbose && println(stderr, "error_function> config: $conf, searchtime: $searchtime, recall: $recall, length: $(length(index))")
        (;
            visited=(minimum(vmin), sum(vacc)/length(knnlist), maximum(vmax)),
            radius=(rmin, ravg, rmax),
            recall=recall,
            searchtime=searchtime/length(knnlist)
        )
    end
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize!(
        index::AbstractSearchContext,
        kind::ErrorFunction=ParetoRecall(),
        space::AbstractSolutionSpace=optimization_space(index);
        queries=nothing,
        ksearch=10,
        numqueries=64,
        initialpopulation=8,
        minbatch=0,
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
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `space`: defines the search space
- `params`: the parameters of the solver, see [`search_models` function from `SearchModels.jl`](https://github.com/sadit/SearchModels.jl) package for more information.
- `verbose`: controls if the procedure is verbose or not
"""
function optimize!(
            index::AbstractSearchContext,
            kind::ErrorFunction=ParetoRecall(),
            space::AbstractSolutionSpace=optimization_space(index);
            queries=nothing,
            ksearch=10,
            numqueries=64,
            initialpopulation=16,
            minbatch=0,
            verbose=false,
            params=SearchParams(; maxpopulation=16, bsize=4, mutbsize=16, crossbsize=8, tol=-1.0, maxiters=16, verbose)
    )

    if queries === nothing
        sample = rand(1:length(index), numqueries) |> unique
        queries = SubDatabase(index.db, sample)
    end

    knnlist = [KnnResult(ksearch) for _ in eachindex(queries)]
    gold = nothing
    if kind isa ParetoRecall || kind isa MinRecall
        db = @view index.db[1:length(index)]
        seq = ExhaustiveSearch(index.dist, db)
        searchbatch(seq, queries, knnlist; minbatch)
        gold = [Set(res.id) for res in knnlist]
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

    errorfun = create_error_function(index, gold, knnlist, queries, ksearch, verbose)

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