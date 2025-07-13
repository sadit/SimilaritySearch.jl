# This file is a part of SimilaritySearch.jl

"""
    Neighborhood(filter::NeighborhoodFilter;
        logbase=2,
        minsize=2)

Convenience constructor, see Neighborhood struct.
"""
function Neighborhood(filter::NeighborhoodFilter;
    logbase=2,
    minsize=2)
    Neighborhood(; logbase, minsize, filter)
end

Base.copy(N::Neighborhood;
        logbase=N.logbase, minsize=N.minsize,
        filter=copy(N.filter)
    ) = Neighborhood(; logbase, minsize, filter)

function neighborhoodsize(N::Neighborhood, n::Integer)::Int
    n == 0 ? 0 : ceil(Int, N.minsize + log(N.logbase, n))
end

"""
    find_neighborhood(copy_, index::SearchGraph{T}, ctx, item; hints=index.hints)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).
The `copy_` function forces to control how the returned KnnResult object is handled because it uses a cache result set from
the given context. 

# Arguments
- `copy_`: A copying function, it controls what is retrieved by the function.
- `index`: The search index.
- `item`: The item to be inserted.
- `ctx`: context, neighborhood, and cache objects to be used
- `hints`: Search hints
"""
function find_neighborhood(copy_::Function, index::SearchGraph, ctx::SearchGraphContext, item; hints=index.hints)
    ksearch = neighborhoodsize(ctx.neighborhood, length(index))
    res = getiknnresult(ksearch, ctx)
    if ksearch > 0
        res = search(index.algo, index, ctx, item, res, hints)
        res = neighborhoodfilter(ctx.neighborhood.filter, index, ctx, item, sortitems!(res))
    end

    copy_(res)
end

"""
    connect_reverse_links(neighborhood::Neighborhood, adj::abstractadjacencylist, n::integer, neighbors::KnnResult)

Internal function to connect reverse links after an insertion
"""
function connect_reverse_links(::Neighborhood, adj::AbstractAdjacencyList, n::Integer, neighbors)
    #maxnlen = log(neighborhood.logbase, n) 
    @inbounds for id in neighbors
        #nlen = neighbors_length(adj, id)
        add_edge!(adj, id, n)
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
    SatNeighborhood(nndist::Float32=1f-4)

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodFilter
    nndist::Float32
end

SatNeighborhood(; nndist::AbstractFloat=1f-4) = SatNeighborhood(convert(Float32, nndist)) 

"""
    DistalSatNeighborhood()

New items are connected with a small set of items computed with a Distal SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage but in reverse order of distance.
"""
struct DistalSatNeighborhood <: NeighborhoodFilter
    nndist::Float32
end

DistalSatNeighborhood(; nndist::AbstractFloat=1f-4) = DistalSatNeighborhood(convert(Float32, nndist)) 

"""
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodFilter end
Base.copy(::IdentityNeighborhood) = IdentityNeighborhood()

## functions

function neighborhoodfilter(::IdentityNeighborhood, ::SearchGraph, ctx::SearchGraphContext, item, res)
    res
end

"""
    filter(sat::DistalSatNeighborhood, index::SearchGraph, item, res, ctx)

filters `res` using the DistSAT strategy.
"""
@inline function neighborhoodfilter(sat::DistalSatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res)
    hsp = getsatknnresult(length(res), ctx)
    hsp_distal_neighborhood_filter!(hsp, distance(G), database(G), center, res; sat.nndist)
end

@inline function neighborhoodfilter(sat::SatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res)
    hsp = getsatknnresult(length(res), ctx)
    hsp_proximal_neighborhood_filter!(hsp, distance(G), database(G), center, res; sat.nndist)
end

## prunning neighborhood
#=

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

=#
