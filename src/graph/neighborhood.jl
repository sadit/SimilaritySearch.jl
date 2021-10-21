# This file is a part of SimilaritySearch.jl

"""
    find_neighborhood(index::SearchGraph{T}, item; res=index.res)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be
its neighbors (intenal function)
"""
function find_neighborhood(index::SearchGraph, item; res=index.res)
    n = length(index)
    N = index.neighborhood
    
    if n > 0
        empty!(res, N.ksearch)
        reduce(N.reduce, search(index, item, res), index)
    else
        KnnResult(N.ksearch)
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
    n = length(index)
    k = index.neighborhood.k

    @inbounds for (id, dist) in neighbors
        v = index.links[id]
        v.k = max(maxlength(v), k) # adjusting maximum size to the current allowed neighborhood size
        push!(v, n => dist)
    end

    push!(index.locks, ReentrantLock())
    apply_callbacks && callbacks(index)

    if index.verbose && length(index) % 100_000 == 0
        println(stderr, "added n=$(length(index)), neighborhood=$(length(neighbors)), $(string(index.search_algo)), $(Dates.now())")
    end
end


"""
    SatNeighborhood(k=32)

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are reduced to a small neighborhood due to the SAT partitioning stage.
"""
mutable struct SatNeighborhood <: NeighborhoodReduction
    near::KnnResult{Int32,Float32}
    SatNeighborhood() = new(KnnResult(1))
end

Base.copy(::SatNeighborhood) = SatNeighborhood()

"""
    reduce(sat::SatNeighborhood, res::KnnResult, index::SearchGraph)

Reduces `res` using the SAT strategy.
"""
function Base.reduce(sat::SatNeighborhood, res::KnnResult, index::SearchGraph)
    near = sat.near
    N = KnnResult(maxlength(res))

    @inbounds for (id, dist) in res
        pobj = index[id]
        empty!(near)
        push!(near, 0, dist)
        for nearID in keys(N)
            d = evaluate(index.dist, index[nearID], pobj)
            push!(near, nearID, d)
        end

        argmin(near) == 0 && push!(N, id => dist)
    end

    N
end

"""
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodReduction end

Base.copy(::IdentityNeighborhood) = IdentityNeighborhood()
Base.reduce(sat::IdentityNeighborhood, res::KnnResult, index::SearchGraph) =  copy(res)
