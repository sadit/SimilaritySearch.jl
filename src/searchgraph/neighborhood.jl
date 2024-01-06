# This file is a part of SimilaritySearch.jl


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
        UInt32[]
    end
end

"""
    connect_reverse_links(adj::abstractadjacencylist, n::integer, neighbors::abstractvector)

Internal function to connect reverse links after an insertion
"""
function connect_reverse_links(adj::AbstractAdjacencyList, n::Integer, neighbors::AbstractVector)
    @inbounds for id in neighbors
        add_edge!(adj, id, n)
    end
end

"""
    connect_reverse_links(adj::AbstractAdjacencyList, sp::Integer, ep::Integer)

Internal function to connect reverse links after an insertion batch
"""
function connect_reverse_links(adj::AbstractAdjacencyList, sp::Integer, ep::Integer)
    @batch minbatch=getminbatch(0, ep-sp+1) per=thread for i in sp:ep
        connect_reverse_links(adj, i, neighbors(adj, i))
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
    dfun = distance(index)
    dist = dist < 0f0 ? evaluate(dfun, item, obj) : dist
    push_item!(near, zero(UInt32), dist)

    @inbounds for linkID in sat_neighborhood
        d = evaluate(dfun, database(index, linkID), obj)
        push_item!(near, linkID, d)
    end

    argmin(near) == zero(Int32)
end

function neighborhoodreduce(::IdentityNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools)
    [item.id for item in res]
end

"""
    reduce(sat::DistalSatNeighborhood, index::SearchGraph, item, res, pools)

Reduces `res` using the DistSAT strategy.
"""
@inline function neighborhoodreduce(::DistalSatNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools, N=UInt32[])
    push!(N, argmax(res))

    @inbounds for i in length(res)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        p = res[i]
        sat_should_push(N, index, item, p.id, p.weight, getsatknnresult(pools)) && push!(N, p.id)
    end

    N
end

@inline function neighborhoodreduce(::SatNeighborhood, index::SearchGraph, item, res, pools::SearchGraphPools, N=UInt32[])
    push!(N, argmin(res))

    @inbounds for i in 2:length(res)
        p = res[i]
        sat_should_push(N, index, item, p.id, p.weight, getsatknnresult(pools)) && push!(N, p.id)
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

    @batch minbatch=minbatch per=thread for i in 1:n
        @inbounds L = neighbors(index.adj, i)
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
    dist = distance(index)
    Threads.@threads for i in eachindex(index.adj)
        @inbounds L = neighbors(index.adj, i)
        if length(L) > r.k
            res = getknnresult(r.k, pools)
            @inbounds c = database(index, i)
            @inbounds for objID in L
                push_item!(res, objID, evaluate(dist, c, database(index, objID)))
            end

            resize!(L, length(res))
            for i in eachindex(res)
                @inbounds L[i] = res[i].id
            end
        end
    end
end

"""
    prune!(r::SatPruning, index::SearchGraph; pools=getpools(index))

Select the SatNeighborhood or DistalSatNeighborhood from available neihghbors
"""
function prune!(r::SatPruning, index::SearchGraph; pools=getpools(index))
    dist = distance(index)
    
    Threads.@threads for i in eachindex(index.adj)
        @inbounds L = neighbors(index.adj, i)
        if length(L) > r.k
            res = getknnresult(length(L), pools)
            @inbounds c = database(index, i)
            @inbounds for objID in L
                push_item!(res, objID, evaluate(dist, c, database(index, objID)))
            end
           
            empty!(L)
            neighborhoodreduce(r.kind, index, c, res, pools, L)
        end
    end
end
