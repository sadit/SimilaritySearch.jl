# This file is a part of SimilaritySearch.jl


function neighborhoodsize(N::Neighborhood, n::Integer)::Int
    n == 0 ? 0 : ceil(Int, N.minsize + log(N.logbase, n))
end

"""
    find_neighborhood(index::SearchGraph{T}, ctx, item, blockrange=1:-1; hints=index.hints)

Searches for `item` neighborhood in the index, i.e., if `item` were in the index whose items should be its neighbors (intenal function).

# Arguments
- `index`: The search index.
- `item`: The item to be inserted.
- `blockrange`: Extra block range for parallel insertions, defaults to an empty range
- `ctx`: context, neighborhood, and cache objects to be used
- `hints`: Search hints
"""
function find_neighborhood(index::SearchGraph, ctx::SearchGraphContext, item, blockrange=1:-1; hints=index.hints)
    ksearch = neighborhoodsize(ctx.neighborhood, length(index))
    res = getiknnresult(ksearch, ctx)
    if ksearch > 0
        search(index.algo[], index, ctx, item, res, hints)
        for i in blockrange  # interblock neighbors
            d = evaluate(distance(index), item, database(index, i))
            d <= ctx.neighborhood.neardup && continue  # avoids self reference and nearest dup in the same block for simplicity
            push_item!(res, i, d)
        end

        output = getsatknnresult(length(res), ctx)
        return neighborhoodfilter(ctx.neighborhood.filter, index, ctx, item, sortitems!(res), output)
    else
        return res  # empty set
    end
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
    minbatch = getminbatch(ep - sp + 1, Threads.nthreads(), 0)
    Threads.@threads :static for j in sp:minbatch:ep
        for i in j:min(ep, j + minbatch - 1)
            connect_reverse_links(neighborhood, adj, i, neighbors(adj, i))
        end
    end
end

"""
    SatNeighborhood()

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodFilter end

"""
    DistalSatNeighborhood()

New items are connected with a small set of items computed with a Distal SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage but in reverse order of distance.
"""
struct DistalSatNeighborhood <: NeighborhoodFilter end

"""
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodFilter end

## functions

neighborhoodfilter(::IdentityNeighborhood, ::SearchGraph, ctx::SearchGraphContext, item, res, output) = res

"""
    filter(sat::DistalSatNeighborhood, index::SearchGraph, item, res, ctx)

filters `res` using the DistSAT strategy.
"""
@inline function neighborhoodfilter(sat::DistalSatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    hsp_distal_neighborhood_filter!(output, distance(G), database(G), center, res; ctx.neighborhood.neardup)
end

@inline function neighborhoodfilter(sat::SatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    hsp_proximal_neighborhood_filter!(output, distance(G), database(G), center, res; ctx.neighborhood.neardup)
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
@kwdef struct SatPruning <: NeighborhoodPruning
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
