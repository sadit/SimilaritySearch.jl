# This file is a part of SimilaritySearch.jl


"""
    Neighborhood(filter::NeighborhoodFilter;
        logbase=2,
        minsize=2, 
        connect_reverse_links_factor=0.8,
        connect_reverse_links_min_factor=0.01)

Convenience constructor, see Neighborhood struct.
"""
function Neighborhood(filter::NeighborhoodFilter;
    logbase=2,
    minsize=2,
    connect_reverse_links_factor=0.8)
    Neighborhood(; logbase, minsize, connect_reverse_links_factor, filter)
end

Base.copy(N::Neighborhood;
        logbase=N.logbase, minsize=N.minsize,
        connect_reverse_links_factor=N.connect_reverse_links_factor,
        filter=copy(N.filter)
    ) = Neighborhood(; logbase, minsize, connect_reverse_links_factor, filter)

neighborhoodsize(N::Neighborhood, index::SearchGraph) = ceil(Int, N.minsize + log(N.logbase, length(index)))

"""
    find_neighborhood(index::SearchGraph{T}, context, item; hints=index.hints)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).
`res` is always reused since `filter` creates a new KnnResult from it (a copy if `filter` in its simpler terms)

# Arguments
- `index`: The search index.
- `item`: The item to be inserted.
- `context`: context, neighborhood, and cache objects to be used
- `hints`: Search hints
"""
function find_neighborhood(index::SearchGraph, context::SearchGraphContext, item; hints=index.hints)
    n = length(index)
    if n > 0
        neighborhood = context.neighborhood
        ksearch = neighborhoodsize(neighborhood, index)
        res = getknnresult(ksearch, context)
        search(index.search_algo, index, context, item, res, hints)
        neighborhoodfilter(neighborhood.filter, index, context, item, res)
    else
        UInt32[]
    end
end

"""
    connect_reverse_links(neighborhood::Neighborhood, adj::abstractadjacencylist, n::integer, neighbors::abstractvector)

Internal function to connect reverse links after an insertion
"""
function connect_reverse_links(neighborhood::Neighborhood, adj::AbstractAdjacencyList, n::Integer, neighbors::AbstractVector)
    p = 1f0
    @inbounds for id in neighbors
        if rand(Float32) < p
            add_edge!(adj, id, n)
            p *= neighborhood.connect_reverse_links_factor
        end
    end
end

"""
    connect_reverse_links(neighborhood::Neighborhood, adj::AbstractAdjacencyList, sp::Integer, ep::Integer)

Internal function to connect reverse links after an insertion batch
"""
function connect_reverse_links(neighborhood::Neighborhood, adj::AbstractAdjacencyList, sp::Integer, ep::Integer)
    @batch minbatch=getminbatch(0, ep-sp+1) per=thread for i in sp:ep
        connect_reverse_links(neighborhood, adj, i, neighbors(adj, i))
    end
end

"""
    SatNeighborhood(hfactor::Float32=0f0)

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodFilter
    hfactor::Float32
    SatNeighborhood(h::AbstractFloat=0.0) = new(convert(Float32, h)) 
end

"""
    DistalSatNeighborhood()

New items are connected with a small set of items computed with a Distal SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage but in reverse order of distance.
"""
struct DistalSatNeighborhood <: NeighborhoodFilter
    hfactor::Float32
    SatNeighborhood(h::AbstractFloat=0.0) = new(convert(Float32, h)) 
end


"""
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodFilter end
Base.copy(::IdentityNeighborhood) = IdentityNeighborhood()

## functions

function neighborhoodfilter(::IdentityNeighborhood, index::SearchGraph, context::SearchGraphContext, item, res)
    [item.id for item in res]
end

"""
    filter(sat::DistalSatNeighborhood, index::SearchGraph, item, res, context)

filters `res` using the DistSAT strategy.
"""
@inline function neighborhoodfilter(sat::DistalSatNeighborhood, index::SearchGraph, context::SearchGraphContext, item, res)
    dfun = distance(index)
    db = database(index)
    hsp_neighborhood = getsatknnresult(length(res), context)
    push_item!(hsp_neighborhood, argmax(res), maximum(res))

    @inbounds for i in length(res)-1:-1:1  # DistSat => works a little better but produces larger neighborhoods
        p = res[i]
        hsp_should_push(hsp_neighborhood, dfun, db, item, p.id, p.weight, sat.hfactor) && push_item!(hsp_neighborhood, p.id, p.weight)
    end

    UInt32.(IdView(hsp_neighborhood))
end

@inline function neighborhoodfilter(sat::SatNeighborhood, index::SearchGraph, context::SearchGraphContext, item, res)
    dfun = distance(index)
    db = database(index)
    hsp_neighborhood = getsatknnresult(length(res), context)
    push_item!(hsp_neighborhood, argmin(res), minimum(res))

    @inbounds for i in 2:length(res)
        p = res[i]
        hsp_should_push(hsp_neighborhood, dfun, db, item, p.id, p.weight, sat.hfactor) && push_item!(hsp_neighborhood, p.id, p.weight)
    end

    UInt32.(IdView(hsp_neighborhood))
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
    SatPruning(k; kind=DistalSatNeighborhood())
    SatPruning(k)

Selects `SatNeighborhood` or `DistalSatNeighborhood` for each vertex. Defaults to `DistalSatNeighborhood`.

- `k`: the threshold size to apply the Sat reduction, i.e., neighbors larger than `k` will be pruned.
"""
@with_kw struct SatPruning <: NeighborhoodPruning
    k::Int
    kind = DistalSatNeighborhood() ## DistalSatNeighborhood, SatNeighborhood    
end

SatPruning(k) = SatPruning(k, DistalSatNeighborhood())

"""
    prune!(r::RandomPruning, index::SearchGraph, context::SearchGraphContext)

Randomly prunes each neighborhood

# Arguments

"""
function prune!(r::RandomPruning, index::SearchGraph, context::SearchGraphContext)
    n = length(index)
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=getminbatch(0, n) per=thread for i in 1:n
        @inbounds L = neighbors(index.adj, i)
        if length(L) > r.k
            shuffle!(L)
            resize!(L, r.k)
        end
    end
end

"""
    prune!(r::KeepNearestPruning, index::SearchGraph, context::SearchGraphContext)

Selects `k` nearest neighbors among the available neighbors
"""
function prune!(r::KeepNearestPruning, index::SearchGraph, context::SearchGraphContext)
    dist = distance(index)
    Threads.@threads for i in eachindex(index.adj)
        @inbounds L = neighbors(index.adj, i)
        if length(L) > r.k
            res = getknnresult(r.k, context)
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
    prune!(r::SatPruning, index::SearchGraph, context::SearchGraphContext)

Select the SatNeighborhood or DistalSatNeighborhood from available neihghbors
"""
function prune!(r::SatPruning, index::SearchGraph, context::SearchGraphContext)
    dist = distance(index)
    
    @batch minbatch=getminbatch(0, length(index)) per=thread for i in eachindex(index.adj)
        L = neighbors(index.adj, i)
        if length(L) > r.k
            res = getknnresult(length(L), context)
            c = database(index, i)
            for objID in L
                push_item!(res, objID, evaluate(dist, c, database(index, objID)))
            end
           
            empty!(L)
            neighborhoodfilter(r.kind, index, context, c, res, L)
        end
    end
end

