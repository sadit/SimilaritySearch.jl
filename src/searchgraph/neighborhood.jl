# This file is a part of SimilaritySearch.jl

"""
    @with_kw mutable struct Neighborhood
    
Determines the size of the neighborhood, \$k\$ is adjusted as a callback, and it is intended to affect previously inserted vertices.
The neighborhood is designed to consider two components \$k=in+out\$, i.e. _in_coming and _out_going edges for each vertex.
- The \$out\$ size is computed as \$minsize + \\log(logbase, n)\$ where \$n\$ is the current number of indexed elements; this is computed searching
for \$out\$  elements in the current index.
- The \$in\$ size is unbounded.
- reduce is intended to postprocess neighbors (after search process, i.e., once out edges are computed); do not change \$k\$ but always must return a copy of the reduced result set.

Note: Set \$logbase=Inf\$ to obtain a fixed number of \$in\$ nodes; and set \$minsize=0\$ to obtain a pure logarithmic growing neighborhood.

"""
@with_kw struct Neighborhood{Reduction<:NeighborhoodReduction}
    logbase::Float32 = 2
    minsize::Int32 = 2
    reduce::Reduction = SatNeighborhood()
end

"""
    Neighborhood(reduce::NeighborhoodReduction; logbase=2, minsize=2)

Convenience constructor, see Neighborhood struct.
"""
Neighborhood(reduce::NeighborhoodReduction; logbase=2, minsize=2) = Neighborhood(; logbase, minsize, reduce)

Base.copy(N::Neighborhood; logbase=N.logbase, minsize=N.minsize, reduce=copy(N.reduce)) =
    Neighborhood(; logbase, minsize, reduce)

neighborhoodsize(N::Neighborhood, index::SearchGraph) = ceil(Int, N.minsize + log(N.logbase, length(index)))

"""
    find_neighborhood(index::SearchGraph{T}, item, neighborhood, pools; hints=index.hints)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).
`res` is always reused since `reduce` creates a new KnnResult from it (a copy if `reduce` in its simpler terms)

# Arguments
- `index`: The search index.
- `item`: The item to be inserted.
- `neighborhood`: A [`Neighborhood`](@ref) object that describes how to compute item's neighborhood.
- `pools`: Cache pools to be used
- `hints`: Search hints
"""
function find_neighborhood(index::SearchGraph, item, neighborhood::Neighborhood, pools::SearchGraphPools; hints=index.hints)
    n = length(index)
    if n > 0
        ksearch = neighborhoodsize(neighborhood, index)
        res = getknnresult(ksearch, pools)
        search(index.search_algo, index, item, res, hints, pools)
        neighborhoodreduce(neighborhood.reduce, index, item, res, pools)
    else
        Int32[]
    end
end

"""
    push_neighborhood!(index::SearchGraph, item, neighbors, callbacks; push_item=true)

Inserts the object `item` into the index, i.e., creates an edge for each item in `neighbors` (internal function)

# Arguments

- `index`: The search index to be modified.
- `item`: The item that will be inserted.
- `neighbors`: An array of indices that will be connected to the new vertex.
- `callbacks`: A [`SearchGraphCallbacks`] object (callback list) that will be called after some insertions
- `push_item`: Specifies if the item must be inserted into the internal `db` (sometimes is already there like in [`index!`](@ref))
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

function neighborhoodreduce(::IdentityNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools)
    copy(res.id)
end

"""
    reduce(sat::DistalSatNeighborhood, index::SearchGraph, item, res, pools)

Reduces `res` using the DistSAT strategy.
"""
@inline function neighborhoodreduce(::DistalSatNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools, N=Int32[])
    push!(N, argmax(res))

    @inbounds for i in length(res)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        (id, dist) = res[i]
        sat_should_push(N, index, item, id, dist, getsatknnresult(pools)) && push!(N, id)
    end

    N
end

@inline function neighborhoodreduce(::SatNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools, N=Int32[])
    push!(N, argmin(res))

    @inbounds for i in 2:length(res)
        (id, dist) = res[i]
        sat_should_push(N, index, item, id, dist, getsatknnresult(pools)) && push!(N, id)
    end

    N
end

## prunning neighborhood


"""
    NeighborhoodPruning

Abstract data type for neighborhood pruning strategies
"""
abstract type NeighborhoodPruning end

"""
    RandomPruning(k)

Selects `k` random edges for each vertex
"""
struct RandomPruning <: NeighborhoodPruning
    k::Int
end

"""
    KeepNearestPruning(k)

Kept `k` nearest neighbor edges for each vertex
"""
struct KeepNearestPruning <: NeighborhoodPruning
    k::Int
end

"""
    SatPruning(k, satkind)
    SatPruning(k)

Selects `SatNeighborhood` or `DistalSatNeighborhood` for each vertex. Defaults to `DistalSatNeighborhood`.


- `k`: the threshold size to apply the Sat reduction, i.e., neighbors larger than `k` will be pruned.
"""
struct SatPruning{SatKind} <: NeighborhoodPruning
    k::Int
    kind::SatKind  ## DistalSatNeighborhood, SatNeighborhood    
end

SatPruning(k) = SatPruning(k, DistalSatNeighborhood())

"""
    prune!(r::RandomPruning, index::SearchGraph;  minbatch=0, pools=getpools(index))

Randomly prunes each neighborhood

# Arguments


"""
function prune!(r::RandomPruning, index::SearchGraph; minbatch=0, pools=getpools(index))
    n = length(index)
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch for i in 1:n
        @inbounds L = index.links[i]
        if length(L) > r.k
            shuffle!(L)
            resize!(L, r.k)
        end
    end
end

"""
    prune!(r::KeepNearestPruning, index::SearchGraph; pools=getpools(index))

Selects `k` nearest neighbors among the the available neighbors
"""
function prune!(r::KeepNearestPruning, index::SearchGraph; pools=getpools(index))
    Threads.@threads for i in eachindex(index.links)
        @inbounds L = index.links[i]
        if length(L) > r.k
            res = getknnresult(r.k, pools)
            @inbounds c = index[i]
            @inbounds for objID in L
                push!(res, objID, evaluate(index.dist, c, index[objID]))
            end

            resize!(L, length(res))
            copy!(L, idview(res))
        end
    end
end

"""
    prune!(r::SatPruning, index::SearchGraph; pools=getpools(index))

Select the SatNeighborhood or DistalSatNeighborhood from available neihghbors
"""
function prune!(r::SatPruning, index::SearchGraph; pools=getpools(index))
    Threads.@threads for i in eachindex(index.links)
        @inbounds L = index.links[i]
        if length(L) > r.k
            res = getknnresult(length(L), pools)
            @inbounds c = index[i]
            @inbounds for objID in L
                push!(res, objID, evaluate(index.dist, c, index[objID]))
            end
           
            empty!(L)
            neighborhoodreduce(r.kind, index, c, res, pools, L)
        end
    end
end