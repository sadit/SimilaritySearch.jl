# This file is a part of SimilaritySearch.jl
export IdentityNeighborhood, DistalSatNeighborhood, SatNeighborhood, find_neighborhood, push_neighborhood!, NeighborhoodSize

"""
    callback(opt::NeighborhoodSize, index)

SearchGraph's callback for adjusting neighborhood strategy
"""
function callback(opt::NeighborhoodSize, index)
    N = index.neighborhood
    N.ksearch = ceil(Int, N.minsize + log(N.logbase, length(index)))
end

"""
    find_neighborhood(index::SearchGraph{T}, item, res::KnnResult, vstate)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).
`res` is always reused since `reduce` creates a new KnnResult from it (a copy if `reduce` in its simpler terms)
"""
function find_neighborhood(index::SearchGraph, item, res::KnnResult, vstate)
    n = length(index)
    
    if n > 0
        empty!(res, index.neighborhood.ksearch)
        reduce(index.neighborhood.reduce, index, item, search(index, item, res; vstate))
    else
        Int32[]
    end
end

"""
    push_neighborhood!(index::SearchGraph, item, neighbors; push_item=true, apply_callbacks=true)

Inserts the object `item` into the index, i.e., creates an edge from items listed in L and the
vertex created for ìtem` (internal function)
"""
function push_neighborhood!(index::SearchGraph, item, neighbors; push_item=true, apply_callbacks=true)
    push_item && push!(index.db, item)
    push!(index.links, neighbors)
    # push!(index.links, Int32[])
    push!(index.locks, Threads.SpinLock())
    n = length(index)
    @inbounds for id in neighbors
        push!(index.links[id], n)
        # push_neighbor!(index.neighborhood.reduce, index.links[id], index, item, n, -1.0)
    end

    apply_callbacks && callbacks(index)

    if index.verbose && length(index) % 100_000 == 0
        println(stderr, "added n=$(length(index)), neighborhood=$(length(neighbors)), $(string(index.search_algo)), $(Dates.now())")
    end
end

const GlobalSatKnnResult = [KnnResult(1)]
function __init__neighborhood()
    for i in 2:Threads.nthreads()
        push!(GlobalSatKnnResult, KnnResult(1))
    end
end

"""
    SatNeighborhood()

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are reduced to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodReduction end
Base.copy(::SatNeighborhood) = SatNeighborhood()

"""
    DistalSatNeighborhood()

New items are connected with a small set of items computed with a Distal SAT like scheme (**cite**).
It starts with `k` near items that are reduced to a small neighborhood due to the SAT partitioning stage but in reverse order of distance.
"""
struct DistalSatNeighborhood <: NeighborhoodReduction end
Base.copy(::DistalSatNeighborhood) = DistalSatNeighborhood()

"""
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodReduction end
Base.copy(::IdentityNeighborhood) = IdentityNeighborhood()

## functions
function push_neighbor!(::Union{DistalSatNeighborhood,SatNeighborhood}, N, index::SearchGraph, item, id::Integer, dist::AbstractFloat)
    near = GlobalSatKnnResult[Threads.threadid()]
    empty!(near)

    @inbounds obj = index[id]
    dist = dist < 0.0 ? evaluate(index.dist, item, obj) : dist
    push!(near, 0, dist)

    @inbounds for linkID in N
        d = evaluate(index.dist, index[linkID], obj)
        push!(near, linkID, d)
    end

    # argmin(near) == 0 && push!(N, id, dist)
    if argmin(near) == 0
        push!(N, id)
        true
    else
        false
    end
end

function push_neighbor!(::IdentityNeighborhood, N, index::SearchGraph, item, id::Integer, dist::AbstractFloat)
    push!(N, id)
    true
end


function Base.reduce(nreduction::NeighborhoodReduction, index::SearchGraph, item, res::KnnResult, N=Int32[])
    for (id, dist) in res
        push_neighbor!(nreduction, N, index, item, id, dist)
    end

    N
end


"""
    reduce(sat::DistalSatNeighborhood, index::SearchGraph, item, res::KnnResult)

Reduces `res` using the DistSAT strategy.
"""
@inline function Base.reduce(sat::DistalSatNeighborhood, index::SearchGraph, item, res::KnnResult, N=Int32[])

    @inbounds for i in lastindex(res):-1:firstindex(res)  # DistSat => works a little better but also produces a bit larger neighborhoods
        id, dist = res[i]
        push_neighbor!(sat, N, index, item, id, dist)
    end

    N
end