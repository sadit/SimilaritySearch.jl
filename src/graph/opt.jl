# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SearchModels, Random
import SearchModels: combine, mutate, config_type

@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 1:4:16
    origin::BeamSearch
end

function _create_bs(o::BeamSearch, bsize)
    BeamSearch(hints=o.hints, bsize=bsize, beam=o.beam, vstate=o.vstate)
end


Base.eltype(::BeamSearchSpace) = BeamSearch
Base.rand(space::BeamSearchSpace) = _create_bs(space.origin, rand(space.bsize))
combine(a::BeamSearch, b::BeamSearch) = _create_bs(a, div(a.bsize + b.bsize, 2))
mutate(space::BeamSearchSpace, c::BeamSearch, iter) = _create_bs(c, SearchModels.scale(c.bsize, s=1.1, lower=1))

@with_kw struct IHCSearchSpace <: AbstractSolutionSpace
    restarts = 1:4:16
    origin::IHCSearch
end

function _create_ihc(o::IHCSearch, restarts)
    IHCSearch(hints=o.hints, restarts=restarts, localimprovements=o.localimprovements, vstate=o.vstate)
end

Base.eltype(::IHCSearchSpace) = IHCSearch
Base.rand(space::IHCSearchSpace) = _create_ihc(space.origin, rand(space.restarts))
combine(a::IHCSearch, b::IHCSearch) = _create_ihc(a, div(a.restarts + b.restarts, 2))
mutate(space::IHCSearchSpace, c::IHCSearch, iter) = _create_ihc(c, SearchModels.scale(c.restarts, s=1.1, lower=1))

"""
    callback(opt::OptimizeParametersCallback, index)

SearchGraph's callback for adjunting search parameters
"""
function callback(opt::OptimizeParametersCallback, index)
    sample = unique(rand(1:length(index), opt.numqueries))
    queries = index[sample]

    error_function = if opt.error === :recall
        seq = ExhaustiveSearch(index.dist, index.db; ksearch=opt.ksearch)
        perf = Performance(seq, queries, opt.ksearch; popnearest=true)
        function error_function1(c)
            copy!(index.search_algo, c)
            p = probe(perf, index)
            index.verbose && println(stderr, "== SearchGraph optimizing recall, err: ", p, ", using configuration: ", c)
            1 / (p.macrorecall + 1)
        end
    elseif opt.error === :distance
        function error_function2(c)
            d = 0.0
            copy!(index.search_algo, c)
            for q in queries
                d += maximum(keys(search(index, q, opt.ksearch)))
            end
            d /= length(queries)

            index.verbose && println(stderr, "== SearchGraph optimizing covering radius, err: ", d, ", using configuration: ", c)
            d
        end
    elseif opt.error == :distance_and_searchtime
        function error_function3(c)
            d = 0.0
            copy!(index.search_algo, c)
            t = time()
            for q in queries
                d += maximum(search(index, q, opt.ksearch))
            end
            t = time() - t
            d *= t
            index.verbose && println(stderr, "== SearchGraph optimizing distance and searchtime, err: ", d, " using configuration: ", c)
            d
        end
    else
        error("unknown $(opt.error), valid options are :recall, :distance, and :distance_and_searchtime")
    end
    
    space = index.search_algo isa BeamSearch ? BeamSearchSpace(origin=index.search_algo) : IHCSearchSpace(origin=index.search_algo)
    bestlist = search_models(space, error_function, opt.initialpopulation; maxpopulation=opt.maxpopulation, maxiters=opt.maxiters, tol=opt.tol)
    config, err = bestlist[1]
    copy!(index.search_algo, config)
    index.verbose && println(stderr, "== finished optimization SearchGraphdistance, err: ", err, ", with configuration: ", config)
    index
end