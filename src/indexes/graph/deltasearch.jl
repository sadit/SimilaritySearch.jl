export DeltaSearch

struct DeltaSearch <: LocalSearchAlgorithm
    montecarlo_size::Int
    delta::Float32
end

DeltaSearch() = DeltaSearch(1, 2.0)
DeltaSearch(other::DeltaSearch) = DeltaSearch(other.montecarlo_size, other.delta)

### local search algorithm

function delta_search(bsearch::DeltaSearch, index::LocalSearchIndex{T}, q::T, res::Result, tabu::MemoryType) where {T, MemoryType}
    # first beam
    beam = SlugKnnResult(Int(bsearch.delta * maxlength(res)))
    if isnull(index.options.oracle)
        estimate_knearest(index.db, index.dist, maxlength(beam), bsearch.montecarlo_size, q, tabu, beam)
    else
        estimate_from_oracle(index, q, beam, tabu, res, get(index.options.oracle))
    end

    cov = -1.0
    done = falses(length(tabu))
    while last(beam).dist != cov
        cov = last(beam).dist
        for node in beam
            if done[node.objID]
                continue
            end
            done[node.objID] = true
            @inbounds for childID in index.links[node.objID]
                if !tabu[childID]
                    tabu[childID] = true
                    d = convert(Float32, index.dist(index.db[childID], q))
                    if d <= cov
                        push!(beam, childID, d)
                    end
                end
            end
        end
    end

    for item in beam
        push!(res, item.objID, item.dist)
    end

    return res
end

function search(bsearch::DeltaSearch, index::LocalSearchIndex{T}, q::T, res::Result) where {T}
    length(index.db) == 0 && return res
    tabu = falses(length(index.db))
    delta_search(bsearch, index, q, res, tabu)
    return res
end

function opt_create_random_state(algo::DeltaSearch, max_value)
    a = max(1, rand() * max_value |> round |> Int)
    b = max(1, rand() * max_value |> round |> Int)
    return DeltaSearch(a, b)
end

function opt_expand_neighborhood(fun, gsearch::DeltaSearch, n::Int, iter::Int)
    f(x, w) = max(1, x + w)
    g(x) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))

    if iter == 1
        for i in 1:8
            opt_create_random_state(gsearch, ceil(Int, log2(n))) |> fun
        end

        DeltaSearch(gsearch.montecarlo_size |> g, gsearch.delta |> g) |> fun
        DeltaSearch(gsearch.montecarlo_size |> g, gsearch.delta) |> fun
        DeltaSearch(gsearch.montecarlo_size, gsearch.delta |> g) |> fun
    end

    w = 2
    while w <= div(32,iter)
        DeltaSearch(f(gsearch.montecarlo_size,  w), gsearch.delta) |> fun
        DeltaSearch(f(gsearch.montecarlo_size, -w), gsearch.delta) |> fun
        DeltaSearch(gsearch.montecarlo_size, f(gsearch.delta,  w)) |> fun
        DeltaSearch(gsearch.montecarlo_size, f(gsearch.delta, -w)) |> fun
        w += w
    end
end
