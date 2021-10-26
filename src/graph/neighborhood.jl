# This file is a part of SimilaritySearch.jl

"""
    find_neighborhood(index::SearchGraph{T}, item, res, vstate)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).
`res` is always reused since `reduce` creates a new KnnResult from it (a copy if `reduce` in its simpler terms)
"""
function find_neighborhood(index::SearchGraph, item, res, vstate)
    n = length(index)
    
    if n > 0
        empty!(res, index.neighborhood.ksearch)
        empty!(vstate)
        reduce(index.neighborhood.reduce, search(index, item, res; vstate), index)
    else
        KnnResult(index.neighborhood.ksearch)
    end
end

"""
    push_neighborhood!(index::SearchGraph, item, neighbors::KnnResult; apply_callbacks=true)

Inserts the object `item` into the index, i.e., creates an edge from items listed in L and the
vertex created for ìtem` (internal function)
"""
function push_neighborhood!(index::SearchGraph, item, neighbors::KnnResult; apply_callbacks=true)
    push!(index.db, item)
    push!(index.links, neighbors)
    push!(index.locks, Threads.SpinLock())
    n = length(index)
    k = index.neighborhood.k

    @inbounds for (id, dist) in neighbors
        vertex = index.links[id]
        vertex.k = max(maxlength(vertex), k) # adjusting maximum size to the current allowed neighborhood size
        push!(vertex, n, dist)
    end

    apply_callbacks && callbacks(index)

    if index.verbose && length(index) % 100_000 == 0
        println(stderr, "added n=$(length(index)), neighborhood=$(length(neighbors)), $(string(index.search_algo)), $(Dates.now())")
    end
end


const GlobalSatKnnResult = [KnnResult(1)]

"""
    SatNeighborhood()

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are reduced to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodReduction end

Base.copy(::SatNeighborhood) = SatNeighborhood()

"""
    reduce(sat::SatNeighborhood, res::KnnResult, index::SearchGraph)

Reduces `res` using the SAT strategy.
"""
function Base.reduce(sat::SatNeighborhood, res::KnnResult, index::SearchGraph)
    near = GlobalSatKnnResult[Threads.threadid()]
    N = KnnResult(ceil(Int, maxlength(res)/2))

    #@inbounds for i in lastindex(res):-1:firstindex(res)  # DistSat => works a little better but also produces a bit larger neighborhoods
    #id, dist = res[i]
    @inbounds for (id, dist) in res
        pobj = index[id]
        empty!(near)
        push!(near, 0, dist)
        for nearID in keys(N)
            d = evaluate(index.dist, index[nearID], pobj)
            push!(near, nearID, d)
        end

        argmin(near) == 0 && push!(N, id, dist)
    end

    N
end

"""
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodReduction end

Base.copy(::IdentityNeighborhood) = IdentityNeighborhood()
Base.reduce(sat::IdentityNeighborhood, res::KnnResult, index::SearchGraph) = copy(res)
