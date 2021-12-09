# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize!, BeamSearchSpace


@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 8:8:64
    Δ = [0.8, 0.9, 1.0, 1.1]                  # this really depends on the dataset, be careful
    bsize_scale = (s=1.5, p1=0.8, p2=0.8, lower=2, upper=512)  # all these are reasonably values
    Δ_scale = (s=1.07, p1=0.8, p2=0.8, lower=0.6, upper=2.0)  # that should work in most datasets
end

Base.hash(c::BeamSearch) = hash((c.bsize, c.Δ, c.maxvisits))
Base.isequal(a::BeamSearch, b::BeamSearch) = a.bsize == b.bsize && a.Δ == b.Δ && a.maxvisits == b.maxvisits
Base.eltype(::BeamSearchSpace) = BeamSearch
Base.rand(space::BeamSearchSpace) = BeamSearch(rand(space.bsize), rand(space.Δ), typemax(Int))

function combine(a::BeamSearch, b::BeamSearch)
    bsize = ceil(Int, (a.bsize + b.bsize) / 2)
    Δ = (a.Δ + b.Δ) / 2
    BeamSearch(; bsize, Δ)
end

function mutate(space::BeamSearchSpace, c::BeamSearch, iter)
    bsize = SearchModels.scale(c.bsize; space.bsize_scale...)
    Δ = SearchModels.scale(c.Δ; space.Δ_scale...)
    BeamSearch(; bsize, Δ)
end

@with_kw mutable struct OptimizeParameters <: Callback
    kind = :pareto_distance_searchtime # :pareto_distance_searchtime, :pareto_recall_searchtime, :minimum_recall_searchtime
    initialpopulation = 16
    params = SearchParams(maxpopulation=16, bsize=4, mutbsize=16, crossbsize=8, tol=-1, maxiters=16)
    ksearch::Int32 = 10
    numqueries::Int32 = 64
    minrecall = 0.9  # used with :minimum_recall_searchtime
    space::BeamSearchSpace = BeamSearchSpace()
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

function eval_beamsearch_config(index::SearchGraph, gold, knn, queries, opt::OptimizeParameters, maxvisits, verbose)
    n = length(index)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)

    @info "--- setting $maxvisits for n=$n --"
    function lossfun(conf)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vacc .= 0.0

        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            res, v = search(conf, index, queries[i], knn[i], index.hints, getvisitedvertices(index); maxvisits)
            ti = Threads.threadid()
            vmin[ti] = min(v, vmin[ti])
            vmax[ti] = max(v, vmax[ti])
            vacc[ti] += v
        end

        v = minimum(vmin), sum(vacc)/length(knn), maximum(vmax)
        rmin, rmax = extrema(maximum, knn)
        ravg = mean(maximum, knn)

        recall_ = if gold !== nothing
            mean(recall_score(gold[i], knn[i]) for i in eachindex(knn))
        else
            nothing
        end

        verbose && println(stderr, "eval_beamsearch_config> config: $conf, opt: $opt, searchtime: $searchtime, recall: $recall_")

        (visited=v, radius=(rmin, ravg, rmax), recall=recall_, searchtime=searchtime/length(knn))
    end
end

"""
    callback(opt::OptimizeParameters, index::SearchGraph)

SearchGraph's callback for adjunting search parameters
"""
function callback(opt::OptimizeParameters, index::SearchGraph)
    optimize!(index, opt; scalemaxvisits=2.0)
end

"""
    optimize!(
        index::SearchGraph,
        opt::OptimizeParameters;
        queries=nothing,
        scalemaxvisits=1.0,
        maxvisits=2 * index.search_algo.maxvisits,
        verbose=index.verbose
    )

Optimizes the index using the `opt` parameters. If `queries=nothing` then it selects a small sample randomply from 
the already indexed objects; this sample size and the number of neighbors are also parameters in `opt`.

Note:
The `visits` parameters is a factor allowing early stopping based on the number of distance evaluations seen in the optimization procedure; please note that in some cases this will reduce the quality to limit the search times.
Please also take into account that inserting items after limiting visits may also cause severe quality degradation when maxvisits is not also updated as required. You can always adjust maxvisits modifying `index.search_algo.maxvisits`.
"""
function optimize!(
        index::SearchGraph,
        opt::OptimizeParameters;
        queries=nothing,
        scalemaxvisits=1.0,
        maxvisits=2 * index.search_algo.maxvisits,
        verbose=index.verbose
    )
    @assert index.search_algo isa BeamSearch
    if queries === nothing
        sample = unique(rand(1:length(index), opt.numqueries))
        queries = index[sample]
    end

    recall_options = (:pareto_recall_searchtime, :minimum_recall_searchtime)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    gold = if opt.kind in recall_options
        db = @view index.db[1:length(index)]
        seq = ExhaustiveSearch(index.dist, db)
        searchbatch(seq, queries, knn; parallel=true)
        Set.(keys.(knn))
    else
        nothing
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

    error_function = eval_beamsearch_config(index, gold, knn, queries, opt, maxvisits, verbose)

    function geterr(p)
        cost = p.visited[2] / M[]
        if opt.kind === :pareto_recall_searchtime    
            cost^2 + (1.0 - p.recall)^2
        elseif opt.kind === :minimum_recall_searchtime
            p.recall < opt.minrecall ? 3.0 - 2 * p.recall : cost
        else
            _kfun(cost) + _kfun(p.radius[2] / R[])
        end
    end
    
    bestlist = search_models(
        error_function,
        opt.space,
        opt.initialpopulation,
        opt.params;
        inspect_population=inspect_pop,
        geterr=geterr)
    config, perf = bestlist[1]
    verbose && println(stderr, "== finished opt. BeamSearch: search-params: $(opt.params), opt-config: $config, perf: $perf, maxvisits=$maxvisits")
    index.search_algo.Δ = config.Δ
    index.search_algo.bsize = config.bsize
    index.search_algo.maxvisits = ceil(Int, scalemaxvisits * perf.visited[end])
    bestlist
end
