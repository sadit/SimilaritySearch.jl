# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize!, BeamSearchSpace


@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 8:8:64
    Δ = [0.8, 0.9, 1.0, 1.1]                  # this really depends on the dataset, be careful
    bsize_scale = (s=1.5, p1=0.8, p2=0.8, lower=2, upper=256)  # all these are reasonably values
    Δ_scale = (s=1.07, p1=0.8, p2=0.8, lower=0.6, upper=1.99)  # that should work in most datasets
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
    maxvisits = n->3log2(n)^3 # will be computed as ceil(Int, maxvisits(length(index)))
    space::BeamSearchSpace = BeamSearchSpace()
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

function recall_searchtime(index::SearchGraph, db, queries, opt::OptimizeParameters, verbose)
    @assert opt.kind in (:pareto_recall_searchtime, :minimum_recall_searchtime)
    seq = ExhaustiveSearch(index.dist, db)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    gtime = @elapsed searchbatch(seq, queries, knn; parallel=true)
    gold = Set.(keys.(knn))
    n = length(index)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    maxvisits = ceil(Int, opt.maxvisits(n))
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
        recall_ = mean(recall(gold[i], knn[i]) for i in eachindex(knn))
        verbose && println(stderr, "pareto_recall_searchtime> config: $conf, opt: $opt, searchtime: $searchtime, recall: $recall_")

        cost = v[2] / maxvisits
        err = if opt.kind === :pareto_recall_searchtime
            cost^2 + (1.0 - recall_)^2
        else # opt.kind === :minimum_recall_searchtime
            recall_ < opt.minrecall ? 3.0 - 2 * recall_ : cost
        end

        (err=err, visited=v, recall=recall_, cost=cost, searchtime=searchtime/length(knn))
    end
end

function pareto_distance_searchtime(index::SearchGraph, queries, opt::OptimizeParameters, verbose)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    n = length(index)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vmean = Vector{Float64}(undef, nt)
    maxvisits = ceil(Int, opt.maxvisits(length(index)))
    rmax = Float64[]

    function lossfun(c)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vmean .= 0

        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            res, v = search(c, index, queries[i], knn[i], index.hints, getvisitedvertices(index); maxvisits)
            ti = Threads.threadid()
            vmin[ti] = min(v, vmin[ti])
            vmax[ti] = max(v, vmax[ti])
            vmean[ti] += v
        end

        v = minimum(vmin), sum(vmean) / length(knn), maximum(vmax)

        ravg = 0.0
        for res in knn
            ravg += maximum(res)
        end
        ravg = ravg / length(knn)

        rmax_ = if length(rmax) == 0
            rmax_ = 0.0
            for res in knn
                rmax_ = max(rmax_, maximum(res))
            end
    
            push!(rmax, rmax_)
            rmax_
        else
            rmax[1]
        end

        cost = v[2] / maxvisits
        verbose && println(stderr, "pareto_distance_searchtime> config: $(c), searchtime: $searchtime, ravg: $ravg, rmax: $rmax_")
        err = _kfun(cost) + _kfun(ravg / rmax_)

        (err=err, visited=v, cost=cost, searchtime=searchtime/length(knn))
    end
end

"""
    callback(opt::OptimizeParameters, index::SearchGraph)

SearchGraph's callback for adjunting search parameters
"""
function callback(opt::OptimizeParameters, index::SearchGraph)
    optimize!(index, opt) 
end

"""
    optimize!(index::SearchGraph, opt::OptimizeParameters; queries=nothing, verbose=index.verbose, visits=2.0)

Optimizes the index using the `opt` parameters. If `queries=nothing` then it selects a small sample randomply from 
the already indexed objects; this sample size and the number of neighbors are also parameters in `opt`.

Note:
The `visits` parameters is a factor allowing early stopping based on the number of distance evaluations seen in the optimization procedure; please note that in some cases this will reduce the quality to limit the search times.
Please also take into account that inserting items after limiting visits may also cause severe quality degradation when maxvisits is not also updated as required. You can always adjust maxvisits modifying `index.search_Algo.maxvisits`.
"""
function optimize!(index::SearchGraph, opt::OptimizeParameters; queries=nothing, verbose=index.verbose, visits=2.0)
    @assert index.search_algo isa BeamSearch
    if queries === nothing
        sample = unique(rand(1:length(index), opt.numqueries))
        queries = index[sample]
    end

    ValidOptions = (:pareto_recall_searchtime, :minimum_recall_searchtime)
    error_function = if opt.kind in ValidOptions
        db = @view index.db[1:length(index)]
        recall_searchtime(index, db, queries, opt, verbose)
    elseif opt.kind === :pareto_distance_searchtime
        pareto_distance_searchtime(index, queries, opt, verbose)
    else
        error("unknown optimization $(opt.kind) for BeamSearch, valid options are $ValidOptions")
    end

    bestlist = search_models(error_function, opt.space, opt.initialpopulation, opt.params; geterr=p->p.err)
    config, perf = bestlist[1]
    verbose && println(stderr, "== finished opt. BeamSearch: search-params: $(opt.params), opt-config: $config, perf: $perf")
    index.search_algo.Δ = config.Δ
    index.search_algo.bsize = config.bsize
    index.search_algo.maxvisits = ceil(Int, visits * perf.visited[end])
    bestlist
end
