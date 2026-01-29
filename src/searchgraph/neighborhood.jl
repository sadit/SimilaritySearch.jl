# This file is a part of SimilaritySearch.jl
export Neighborhood, IdentityNeighborhood, DistalSatNeighborhood, SatNeighborhood, KCentersNeighborhood
export find_neighborhood

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
    struct IdentityNeighborhood

It does not modifies the given neighborhood
"""
struct IdentityNeighborhood <: NeighborhoodFilter end

neighborhoodfilter(::IdentityNeighborhood, ::SearchGraph, ctx::SearchGraphContext, item, res, output) = res

"""
    SatNeighborhood()

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage.
"""
struct SatNeighborhood <: NeighborhoodFilter end

@inline function neighborhoodfilter(sat::SatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    hsp_proximal_neighborhood_filter!(output, distance(G), database(G), center, res; ctx.neighborhood.neardup)
end


"""
    DistalSatNeighborhood()

New items are connected with a small set of items computed with a Distal SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage but in reverse order of distance.
"""
struct DistalSatNeighborhood <: NeighborhoodFilter end


"""
    filter(sat::DistalSatNeighborhood, index::SearchGraph, item, res, ctx)

filters `res` using the DistSAT strategy.
"""
@inline function neighborhoodfilter(sat::DistalSatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    hsp_distal_neighborhood_filter!(output, distance(G), database(G), center, res)
end


struct KCentersNeighborhood <: NeighborhoodFilter end

@inline function neighborhoodfilter(N::KCentersNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    S = SubDatabase(database(G), IdView(res))
    k = ceil(Int, log2(length(res)))
    k = min(16, k)
    C = fft(distance(G), S, k; threads=false, verbose=false)
    for i in C.centers
        push_item!(output, res[i])
    end

    output
end

