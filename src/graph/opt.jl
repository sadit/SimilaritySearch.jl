# This file is a part of SimilaritySearch.jl

using SearchModels, Random
import SearchModels: combine, mutate
export OptimizeParameters, optimize!


@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 8:8:64
    Δ = [0.97, 1.0, 1.03]                     # this really depends on the dataset
    bsize_scale = (s=1.5, lower=2, upper=128) # all these are reasonably values
    Δ_scale = (s=1.03, lower=0.7, upper=1.3) # that should work in most datasets
end

Base.eltype(::BeamSearchSpace) = BeamSearch
Base.rand(space::BeamSearchSpace) = BeamSearch(rand(space.bsize), rand(space.Δ))

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
    tol::Float32 = 0.01
    maxiters::Int32 = 8
    space::BeamSearchSpace = BeamSearchSpace()
end

function pareto_recall_searchtime(index::SearchGraph, queries, opt::OptimizeParameters, verbose)
    seq = ExhaustiveSearch(index.dist, index.db)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]
    gtime = @elapsed searchbatch(seq, queries, knn; parallel=true)
    gold = Set.(keys.(knn))

    function lossfun(c)
        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            search(c, index, queries[i], knn[i], index.hints, getvisitedvertices(index))
        end

        recall_ = sum(recall(gold[i], knn[i]) for i in eachindex(knn)) / length(knn)
        verbose && println(stderr, "pareto_recall_searchtime> config: $(c), searchtime: $searchtime, recall: $recall_")
        sqrt((searchtime / gtime)^2 + (1.0 - recall_)^2)
    end
end

function pareto_distance_searchtime(index::SearchGraph, queries, opt::OptimizeParameters, verbose)
    knn = [KnnResult(opt.ksearch) for i in eachindex(queries)]

    function lossfun(c)
        searchtime = @elapsed Threads.@threads for i in eachindex(queries)
            empty!(knn[i], opt.ksearch)
            search(c, index, queries[i], knn[i], index.hints, getvisitedvertices(index))
        end

        dmax_ = sum(maximum(knn_) for knn_ in knn) / length(queries)
        verbose && println(stderr, "pareto_distance_searchtime> config: $(c), searchtime: $searchtime, dmax: $dmax_")
        sqrt(searchtime^2 + dmax_^2)
    end
end


"""
    callback(opt::OptimizeParameters, index::SearchGraph)

SearchGraph's callback for adjunting search parameters
"""
function callback(opt::OptimizeParameters, index::SearchGraph)
    optimize!(index, opt)
end

function optimize!(index::SearchGraph, opt::OptimizeParameters; queries=nothing, verbose=index.verbose)
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

    bestlist = search_models(opt.space, error_function, opt.initialpopulation; maxpopulation=opt.maxpopulation, maxiters=opt.maxiters, tol=opt.tol, verbose=verbose)
    config, err = bestlist[1]
    index.search_algo.Δ = config.Δ
    index.search_algo.bsize = config.bsize
    verbose && println(stderr, "== finished optimization BeamSearch, err: ", err, ", with configuration: ", config, "opt:", opt)
    index
end
