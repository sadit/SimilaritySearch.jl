# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize_index!, MinRecall, OptRadius, ParetoRecall, ParetoRadius

abstract type ErrorFunction end
@with_kw struct MinRecall <: ErrorFunction
    minrecall::Float32 = 0.9f0
end

@with_kw struct OptRadius <: ErrorFunction
    tol::Float32 = 0.1
end

struct ParetoRecall <: ErrorFunction end
struct ParetoRadius <: ErrorFunction end

function runconfig0(conf, index::AbstractSearchIndex, ctx::AbstractContext, queries::AbstractDatabase, i::Integer, res::AbstractKnn)
    runconfig(conf, index, ctx, queries[i], res)
end

function setconfig! end

function create_error_function(index::AbstractSearchIndex, context::AbstractContext, gold, knns, queries, verbose)
    n = length(index)
    m = length(queries)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    cov = Vector{Float64}(undef, m)
    R = [Set{Int32}() for _ in knns]

    function lossfun(conf)
        fill!(vmin, typemax(eltype(vmin)))
        fill!(vmax, typemin(eltype(vmax)))
        fill!(vacc, 0.0)
        empty!(cov)
        
        searchtime = @elapsed begin
            @batch minbatch=getminbatch(0, m) per=thread for i in 1:m
                r = reuse!(knns[i])
                r = runconfig0(conf, index, context, queries, i, r)
                ti = Threads.threadid()
                vmin[ti] = min(r.cost, vmin[ti])
                vmax[ti] = max(r.cost, vmax[ti])
                vacc[ti] += r.cost
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
                union!(R[i], idset(r))
            end

            macrorecall(gold, R)
        else
            nothing
        end

        visited = (min=minimum(vmin), mean=sum(vacc)/m, max=maximum(vmax))
        verbose && println(stderr, "error_function> config: $conf, searchtime: $searchtime per query, recall: $recall, length: $(length(index)), radius: $radius, visited: $visited")
        (; visited, radius, recall, searchtime, conf)
    end
end


_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize_index!(
        index::AbstractSearchIndex,
        context::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        ctx=GenericContext(context),
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
    - `maxiters=16`: maximum number of iterations.
    - `verbose`: controls if the procedure is verbose or not
"""
function optimize_index!(
        index::AbstractSearchIndex,
        context::AbstractContext,
        kind::ErrorFunction=MinRecall(0.9);
        space::AbstractSolutionSpace=optimization_space(index),
        ctx=GenericContext(context),
        queries=nothing,
        ksearch=10,
        numqueries=64,
        initialpopulation=16,
        maxpopulation=16,
        bsize=4,
        mutbsize=16,
        crossbsize=8,
        maxiters=16,
        verbose=false,
        params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, maxiters, verbose)
    )

    db = database(index)
    if queries === nothing
        @info "using $numqueries random queries from the dataset"
        sample = rand(1:length(index), numqueries) |> unique
        queries = SubDatabase(db, sample)
    else
        @info "using $(length(queries)) given as hyperparameter"
    end

    knnsmatrix = zeros(IdWeight, ksearch, length(queries))
    knns = [xknn(c) for c in eachcol(knnsmatrix)]
    gold = nothing
    if kind isa ParetoRecall || kind isa MinRecall
        db = @view db[1:length(index)]
        seq = ExhaustiveSearch(distance(index), db)
        searchbatch!(seq, ctx, queries, knns)
        gold = [idset(viewitems(c)) for c in knns]
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

    getperformance = create_error_function(index, context, gold, knns, queries, verbose)

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
            sort!(view(population, 1:params.bsize), by=p->p.second.visited.mean)
        else
            sort!(population, by=getcost)
        end

        population
    end

    function convergence(curr, prev)
        abs(getcost(prev) - getcost(curr)) <= 1e-3
    end

    bestlist = search_models(getperformance, space, initialpopulation, params; inspect_population, sort_by_best, convergence)
   
    if length(bestlist) == 0
        verbose && println(stderr, "== WARN optimization failure; unable to find usable configurations")
    else
        config, perf = bestlist[1]
        verbose && println(stderr, "== finished opt. $(typeof(index)): search-params: $(params), opt-config: $config, perf: $perf, kind=$(kind), length=$(length(index)), perf=$perf")
        setconfig!(config, index, perf)
    end
    bestlist
end

