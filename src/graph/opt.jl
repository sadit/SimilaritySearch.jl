# This file is a part of SimilaritySearch.jl

using SearchModels, Random
using StatsBase
import SearchModels: combine, mutate
export OptimizeParameters, optimize!


@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 8:8:64
    Δ = [0.97, 1.0, 1.03]                     # this really depends on the dataset
    bsize_scale = (s=1.5, lower=2, upper=128) # all these are reasonably values
    Δ_scale = (s=1.03, lower=0.7, upper=1.3) # that should work in most datasets
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
    space::BeamSearchSpace = BeamSearchSpace()
end

function pareto_recall_searchtime(index::SearchGraph, queries, opt::OptimizeParameters, verbose)
    seq = ExhaustiveSearch(index.dist, index.db)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    gtime = @elapsed searchbatch(seq, queries, knn; parallel=true)
    gold = Set.(keys.(knn))
    visited_ = Vector{Int}(undef, length(knn))
    n = length(index)

    function lossfun(c)
        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            res, v = search(c, index, queries[i], knn[i], index.hints, getvisitedvertices(index))
            visited_[i] = v
        end

        recall_ = mean(recall(gold[i], knn[i]) for i in eachindex(knn))
        v = extrema(visited_)
        verbose && println(stderr, "pareto_recall_searchtime> config: $c, opt: $opt, searchtime: $searchtime, recall: $recall_")

        # err = (searchtime / gtime)^2 + (1.0 - recall_)^2 
        #length(S) == 0 && push!(S, v[end])
        #err = mean(v) / S[1] + (1.0 - recall_)
        err = (mean(v) / n)^2 + (1.0 - recall_)^2
        #err = (1.0 - recall_)

        if opt.minrecall > 0
            err += max(opt.minrecall - recall_, 0.0)
        end

        (err=err, visited=v, recall=recall_, searchtime=searchtime/length(knn))

    end
end

function pareto_distance_searchtime(index::SearchGraph, queries, opt::OptimizeParameters, verbose)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    S = Float64[]
    visited_ = Vector{Int}(undef, length(knn))

    function lossfun(c)
        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            res, v = search(c, index, queries[i], knn[i], index.hints, getvisitedvertices(index))
            visited_[i] = v
        end

        v = extrema(visited_)
        dmax_ = sum(maximum(knn_) for knn_ in knn) / length(queries)
        if length(S) == 0
            push!(S, v[end])
            push!(S, dmax_)
        end

        verbose && println(stderr, "pareto_distance_searchtime> config: $(c), searchtime: $searchtime, dmax: $dmax_")
        err = sqrt((v[end]/S[1])^2 + (dmax_/S[2])^2)

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
function optimize!(index::SearchGraph, opt::OptimizeParameters; queries=nothing, verbose=index.verbose, visits=2.0)
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
