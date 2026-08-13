# This file is a part of SimilaritySearch.jl
export Neighborhood, IdentityNeighborhood, DistalSatNeighborhood, SatNeighborhood, KCentersNeighborhood
export find_neighborhood!
export direct_neighbors, reverse_neighbors, remove_direct_links!, remove_reverse_links!

function neighborhoodsize(N::Neighborhood, n::Integer)::Int
    n == 0 ? N.minsize : ceil(Int, N.minsize + log(N.logbase, n))
end

"""
    find_neighborhood!(out::AbstractKnn, index::SearchGraph, ctx::SearchGraphContext, item, tmp::AbstractKnn, blockrange; hints=index.hints)

Searches for `item`'s neighborhood in the index, i.e., if `item` were in the index, which items should be its neighbors (internal function).

# Arguments
- `out`: `AbstractKnn` object where the resulting (filtered) neighborhood is stored.
- `index`: The search index.
- `ctx`: context, neighborhood, and cache objects to be used.
- `item`: The item to be inserted.
- `tmp`: `AbstractKnn` object used as scratch space for the raw (unfiltered) search results.
- `blockrange`: Extra block range for parallel insertions, defaults to an empty range.

# Keyword Arguments
- `hints`: Search hints
"""
function find_neighborhood!(out::AbstractKnn, index::SearchGraph, ctx::SearchGraphContext, item, tmp::AbstractKnn, blockrange; hints=index.hints)
    n = length(index)
   
    if n > 0
        vstate = getvstate(length(index), ctx)
        search(index.algo[], index, ctx, item, tmp, hints, vstate)
    end

    for i in blockrange  # interblock neighbors
        #@show i => typeof(item) => typeof(database(index, i))
        d = evaluate(distance(index), item, database(index, i))
        d <= ctx.neighborhood.neardup && continue  # avoids self reference and nearest dup in the same block for simplicity
        push_item!(tmp, i, d)
    end

    if length(tmp) > 0 ## only normal on length(blockrange) == 0 && n == 0
        neighborhoodfilter(ctx.neighborhood.filter, index, ctx, item, sortitems!(tmp), out)
    end
    
    out
end

"""
    connect_reverse_links!(adj::AbstractAdjList, nodeID::integer, neighbors::KnnResult)

Internal function to connect reverse links after an insertion
"""
function connect_reverse_links!(adj::AbstractAdjList, nodeID::Integer, neighbors)
    connect_reverse_links!(adj, nodeID, neighbors) do relID
        relID != nodeID    # avoid loops and weird behaviours, i.e., distance functions with d(x, x) != 0)
    end
end

function connect_reverse_links!(mustconnect::Function, adj::AbstractAdjList, nodeID::Integer, neighbors)
    #@info nodeID => reinterpret(Int32, neighbors)
    for relID in neighbors
        mustconnect(relID) && add!(adj, relID, nodeID)
    end
end

"""
    connect_reverse_links!(adj::AbstractAdjList, sp::Integer, ep::Integer; scheduler::Symbol=get_batch_scheduler())

Internal function to connect reverse links after an insertion batch. `scheduler` is the
[`@BATCHES`](@ref) scheduler used for the batch (pass `ctx.scheduler` when called from a
context-bearing caller, since this function has no context of its own); defaults to
[`get_batch_scheduler`](@ref).
"""
function connect_reverse_links!(adj::AbstractAdjList, sp::Integer, ep::Integer; scheduler::Symbol=get_batch_scheduler())
    # The double step algorithm is to avoid weird race conditions
    @BATCHES getminbatch(ep - sp + 1) scheduler=scheduler for nodeID in sp:ep  # connect all elements smaller than sp:ep
        connect_reverse_links!(adj, nodeID, neighbors(adj, nodeID)) do relID
            relID < sp
        end
    end

    L = neighbors_length.(Ref(adj), sp:ep)  # to avoid loop for 'secondary' links
    for (i, nodeID) in enumerate(sp:ep)  # connect all elements smaller than sp:ep
        N = neighbors(adj, nodeID)
        connect_reverse_links!(adj, nodeID, view(N, 1:L[i])) do relID
            sp <= relID && relID != nodeID
            #relID != nodeID
        end
    end
end

"""
    direct_neighbors(g::SearchGraph, i) -> AbstractVector

View of node `i`'s direct neighbors, i.e. `neighbors(g.adj, i)[1:g.directcount[i]]`. See
[`reverse_neighbors`](@ref) for the complementary view, and
[`remove_direct_links!`](@ref)/[`remove_reverse_links!`](@ref) to permanently strip one or the
other.
"""
direct_neighbors(g::SearchGraph, i) = view(neighbors(g.adj, i), 1:g.directcount[i])

"""
    reverse_neighbors(g::SearchGraph, i) -> AbstractVector

View of node `i`'s reverse-inserted neighbors, i.e. `neighbors(g.adj, i)[g.directcount[i]+1:end]`
-- the ones added by [`connect_reverse_links!`](@ref) when some later-inserted node discovered
`i` as one of its own direct neighbors. See [`direct_neighbors`](@ref) for the complementary
view.
"""
reverse_neighbors(g::SearchGraph, i) = view(neighbors(g.adj, i), g.directcount[i]+1:neighbors_length(g.adj, i))

"""
    remove_reverse_links!(g::SearchGraph) -> g

Strips every node's reverse-inserted neighbors (see [`reverse_neighbors`](@ref)), keeping only
direct ones. Intended for testing/experimentation -- e.g. to check whether direct edges alone
are sufficient for navigation. See [`remove_direct_links!`](@ref) for the companion operation.
"""
function remove_reverse_links!(g::SearchGraph)
    for i in eachindex(g.adj)
        resize!(neighbors(g.adj, i), g.directcount[i])
    end
    g
end

"""
    remove_direct_links!(g::SearchGraph) -> g

Strips every node's direct neighbors (see [`direct_neighbors`](@ref)), keeping only
reverse-inserted ones. Intended for testing/experimentation -- e.g. to check whether reverse
edges alone are sufficient for navigation. Nodes that never received a reverse edge become
neighborless (empty). If *no* node anywhere has any reverse-inserted neighbors (e.g. a graph
built via `:knr` static indexing, which never distinguishes direct/reverse, or a graph on which
[`connect_reverse_links!`](@ref) simply never ran), this would empty the entire graph -- a
warning is emitted in that case rather than silently proceeding. See
[`remove_reverse_links!`](@ref) for the companion operation.
"""
function remove_direct_links!(g::SearchGraph)
    if all(g.directcount[i] == neighbors_length(g.adj, i) for i in eachindex(g.adj))
        @warn "remove_direct_links!: no node has any reverse-inserted neighbors (e.g. a :knr-built graph, or connect_reverse_links! never ran) -- this will empty the entire graph"
    end

    for i in eachindex(g.adj)
        deleteat!(neighbors(g.adj, i), 1:g.directcount[i])
        g.directcount[i] = 0
    end
    g
end

"""
    IdentityNeighborhood()

A [`NeighborhoodFilter`](@ref) that does not modify the given neighborhood, i.e., it passes through the candidate result set unchanged.

# Examples

```julia
neighborhood = Neighborhood(filter=IdentityNeighborhood())
```
"""
struct IdentityNeighborhood <: NeighborhoodFilter end

neighborhoodfilter(::IdentityNeighborhood, ::SearchGraph, ctx::SearchGraphContext, item, res, output) = res

"""
    SatNeighborhood()

New items are connected with a small set of items computed with a SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage.

# Examples

```julia
neighborhood = Neighborhood(filter=SatNeighborhood())  # the default filter
```
"""
struct SatNeighborhood <: NeighborhoodFilter end

@inline function neighborhoodfilter(sat::SatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    hsp_proximal_neighborhood_filter!(output, distance(G), database(G), center, res; ctx.neighborhood.neardup)
end


"""
    DistalSatNeighborhood()

New items are connected with a small set of items computed with a Distal SAT like scheme (**cite**).
It starts with `k` near items that are filterd to a small neighborhood due to the SAT partitioning stage but in reverse order of distance.

# Examples

```julia
neighborhood = Neighborhood(filter=DistalSatNeighborhood())
```
"""
struct DistalSatNeighborhood <: NeighborhoodFilter end


"""
    neighborhoodfilter(sat::DistalSatNeighborhood, index::SearchGraph, ctx::SearchGraphContext, item, res, output)

Filters `res` using the DistSAT strategy.
"""
@inline function neighborhoodfilter(sat::DistalSatNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    hsp_distal_neighborhood_filter!(output, distance(G), database(G), center, res)
end


"""
    KCentersNeighborhood()

A [`NeighborhoodFilter`](@ref) that reduces the given candidate neighborhood `res` by
computing a small set of `k`-centers over it (using a farthest-first traversal) and keeping
only the resulting centers, so that the final neighborhood is diverse rather than simply the
closest items.

# Examples

```julia
neighborhood = Neighborhood(filter=KCentersNeighborhood())
```
"""
struct KCentersNeighborhood <: NeighborhoodFilter end

@inline function neighborhoodfilter(N::KCentersNeighborhood, G::SearchGraph, ctx::SearchGraphContext, center, res, output)
    S = SubDatabase(database(G), IdView(res))
    k = ceil(Int, log2(length(res)))
    k = min(16, k)
    C = fft(distance(G), S, k; threads=false, verbose=false, scheduler=:sequential)
    for i in C.centers
        push_item!(output, res[i])
    end

    output
end

