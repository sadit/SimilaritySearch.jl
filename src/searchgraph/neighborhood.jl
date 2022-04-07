# This file is a part of SimilaritySearch.jl

"""
    execute_callback(opt::NeighborhoodSize, index)

SearchGraph's callback for adjusting neighborhood strategy
"""
function execute_callback(opt::NeighborhoodSize, index)
    N = index.neighborhood
    N.ksearch = ceil(Int, N.minsize + log(N.logbase, length(index)))
end

"""
    find_neighborhood(index::SearchGraph{T}, item, pools; hints=index.hints)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).
`res` is always reused since `reduce` creates a new KnnResult from it (a copy if `reduce` in its simpler terms)
"""
function find_neighborhood(index::SearchGraph, item, pools::SearchGraphPools; hints=index.hints)
    n = length(index)
    if n > 0
        res = getknnresult(index.neighborhood.ksearch, pools)
        search(index.search_algo, index, item, res, hints, pools)
        reduce_neighborhood(index.neighborhood.reduce, index, item, res, pools)
    else
        Int32[]
    end
end

"""
    push_neighborhood!(index::SearchGraph, item, neighbors, callbacks; push_item=true)

Inserts the object `item` into the index, i.e., creates an edge from items listed in L and the
vertex created for ìtem` (internal function)
"""
function push_neighborhood!(index::SearchGraph, item, neighbors, callbacks; push_item=true)
    push_item && push!(index.db, item)
    push!(index.links, neighbors)
    push!(index.locks, Threads.SpinLock())
    n = length(index)
    n == 1 && return
    ## vstate = getvisitedvertices(index)
    @inbounds for id in neighbors
        push!(index.links[id], n)  # sat push?
    end

    callbacks !== nothing && execute_callbacks(callbacks, index)

    if index.verbose && length(index) % 100_000 == 0
        println(stderr, "added n=$(length(index)), neighborhood=$(length(neighbors)), $(string(index.search_algo)), $(Dates.now())")
    end
end

const GlobalSatKnnResult = [KnnResult(1)]
function __init__neighborhood()
    for _ in 2:Threads.nthreads()
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

function sat_should_push(sat_neighborhood::T, index, item, id, dist, near::KnnResult) where T
    @inbounds obj = index[id]
    dist = dist < 0.0 ? evaluate(index.dist, item, obj) : dist
    push!(near, 0, dist)

    @inbounds for linkID in sat_neighborhood
        d = evaluate(index.dist, index[linkID], obj)
        push!(near, linkID, d)
    end

    argmin(near) == 0
end

function reduce_neighborhood(::IdentityNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools)
    copy(res.id)
end

"""
    reduce(sat::DistalSatNeighborhood, index::SearchGraph, item, res, pools)

Reduces `res` using the DistSAT strategy.
"""
@inline function reduce_neighborhood(::DistalSatNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools)
    N = Vector{Int32}(undef, 2)
    resize!(N, 0)
    @inbounds for i in length(res):-1:1  # DistSat => works a little better but also produces a bit larger neighborhoods
        id, dist = getpair(res, i)
        sat_should_push(N, index, item, id, dist, getsatknnresult(pools)) && push!(N, id)
    end

    N
end

@inline function reduce_neighborhood(::SatNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools)
    N = Vector{Int32}(undef, 2)
    resize!(N, 0)
    @inbounds for (id, dist) in res  # DistSat => works a little better but also produces a bit larger neighborhoods
        sat_should_push(N, index, item, id, dist, getsatknnresult(pools)) && push!(N, id)
    end

    N
end

