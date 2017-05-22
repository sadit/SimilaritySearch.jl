export GallopingNeighborhood

mutable struct GallopingNeighborhood <: NeighborhoodAlgorithm
    increase_factor::Float32
    decrease_factor::Float32
    lower::Float32
    upper::Float32
    neighborhood::Int
end

function GallopingNeighborhood()
    return GallopingNeighborhood(1.5)
end

function GallopingNeighborhood(factor)
    return GallopingNeighborhood(factor, 1f0/factor, 0.95, 1.05, 1)
end

function optimize_neighborhood!{T}(algo::GallopingNeighborhood, index::LocalSearchIndex{T}, perf, recall)
    # restarts, beam_size, montecarlo_size = index.restarts, index.beam_size, index.montecarlo_size
    # index.restarts, index.beam_size, index.montecarlo_size = 1, 1, 1
    SearchType = typeof(index.search_algo)
    tmp, index.search_algo = index.search_algo, SearchType()
    p = probe(perf, index)
    if p.recall <= recall * algo.lower
        # the recall is too low, it needs larger neighborhoods
        algo.neighborhood = algo.increase_factor * algo.neighborhood |> ceil |> Int
    elseif p.recall >= recall * algo.upper
        # the recall is too high, it needs smaller neighborhoods
        algo.neighborhood = algo.decrease_factor * algo.neighborhood |> ceil |> Int
    end
    n = length(index.db)
    algo.neighborhood = min(max(1, algo.neighborhood), ceil(Int, log2(n)^2))
    index.search_algo = tmp
    # index.restarts, index.beam_size, index.montecarlo_size = restarts, beam_size, montecarlo_size
end

function neighborhood{T}(algo::GallopingNeighborhood, index::LocalSearchIndex{T}, item::T)
    k = algo.neighborhood
    nbuffer = Vector{Int32}(0)
    knn = search(index, item, KnnResult(k))

    for p in knn
        push!(nbuffer, p.objID)
    end

    return knn, nbuffer
end
