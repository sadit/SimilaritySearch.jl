# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize!, BeamSearchSpace


@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 8:8:32
    Δ = [0.97, 1.0, 1.03]                     # this really depends on the dataset
    bsize_scale = (s=1.5, lower=2, upper=256) # all these are reasonably values
    Δ_scale = (s=1.03, lower=0.7, upper=1.9)  # that should work in most datasets
end

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
    kind = :pareto_distance_searchtime # :pareto_distance_searchtime and :pareto_recall_searchtime
    ksearch::Int32 = 10
    numqueries::Int32 = 32
    initialpopulation::Int32 = 8
    maxpopulation::Int32 = 4
    tol::Float32 = 0.001
    maxiters::Int32 = 8
    minrecall = 0.0  # only for :pareto_recall_searchtime
    maxvisits = n->2 * log(2, n)^3  # will be computed as ceil(Int, maxvisits(length(index)))
    space::BeamSearchSpace = BeamSearchSpace()
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

function pareto_recall_searchtime(index::SearchGraph, queries, opt::OptimizeParameters, verbose)
    seq = ExhaustiveSearch(index.dist, index.db)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    gtime = @elapsed searchbatch(seq, queries, knn; parallel=true)
    gold = Set.(keys.(knn))
    n = length(index)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    maxvisits = ceil(Int, opt.maxvisits(length(index)))

    function lossfun(c)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vacc .= 0.0

        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            res, v = search(c, index, queries[i], knn[i], index.hints, getvisitedvertices(index); maxvisits)
            ti = Threads.threadid()
            vmin[ti] = min(v, vmin[ti])
            vmax[ti] = max(v, vmax[ti])
            vacc[ti] += v
        end

        v = minimum(vmin), sum(vacc)/length(knn), maximum(vmax)
        recall_ = mean(recall(gold[i], knn[i]) for i in eachindex(knn))
        verbose && println(stderr, "pareto_recall_searchtime> config: $c, opt: $opt, searchtime: $searchtime, recall: $recall_")

        # err = (searchtime / gtime)^2 + (1.0 - recall_)^2 
        #length(S) == 0 && push!(S, v[end])
        #err = mean(v) / S[1] + (1.0 - recall_)
        err = (v[2] / maxvisits)^2 + (1.0 - recall_)^2 + max(opt.minrecall - recall_, 0.0)^2
        ####err = _kfun(v[2] / maxvisits) + _kfun(1.0 - recall_) + _kfun(max(opt.minrecall - recall_, 0.0))

        (err=err, visited=v, recall=recall_, searchtime=searchtime/length(knn))
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
        
        verbose && println(stderr, "pareto_distance_searchtime> config: $(c), searchtime: $searchtime, ravg: $ravg, rmax: $rmax_")
        err = _kfun(v[2] / maxvisits) + _kfun(ravg / rmax_)
        #err = (mean(v)/n)^2 + (dmax_)^2

        (err=err, visited=v, searchtime=searchtime/length(knn))
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
function optimize!(index::SearchGraph, opt::OptimizeParameters; queries=nothing, verbose=index.verbose, visits=2.0, maxvisits=sqrt)
    @assert index.search_algo isa BeamSearch
    if queries === nothing
        sample = unique(rand(1:length(index), opt.numqueries))
        queries = index[sample]
    end

    error_function = if opt.kind === :pareto_recall_searchtime
        pareto_recall_searchtime(index, queries, opt, verbose)
    elseif opt.kind === :pareto_distance_searchtime
        pareto_distance_searchtime(index, queries, opt, verbose)
    else
        error("unknown optimization $(opt.kind) for BeamSearch, valid options are :pareto_distance_searchtime and :pareto_recall_searchtime")
    end

    params = SearchParams(maxpopulation=opt.maxpopulation, maxiters=opt.maxiters, tol=opt.tol, verbose=verbose)
    bestlist = search_models(error_function, opt.space, opt.initialpopulation, params; geterr=p->p.err)
    config, perf = bestlist[1]
    @show config, perf
    index.search_algo.Δ = config.Δ
    index.search_algo.bsize = config.bsize

    index.search_algo.maxvisits = ceil(Int, visits * perf.visited[end])
    verbose && println(stderr, "== finished optimization BeamSearch, perf: ", perf, ", with configuration: ", config, "opt:", opt)
end
