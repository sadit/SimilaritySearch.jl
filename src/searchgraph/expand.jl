# This file is a part of SimilaritySearch.jl
#
# The second stage of a search over a graph with near-duplicate members: the first stage
# answers with representatives, one per cluster; these give back the raw neighbors.

export expand, expand!

"""
    expand(G, q, res) -> iterator of IdDist
    expand(G, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32})

Iterates the result `res` of a search for `q` -- a queue, or the `(ids, dists)` pair of one
column of [`searchbatch`](@ref SimilaritySearch.searchbatch)'s matrices -- giving each item back followed by the members
of its cluster (see [`Members`](@ref)), each member with its distance to `q` evaluated by
the index's distance. Nothing is modified and nothing is trimmed: a result of `k`
representatives yields `k` items plus every member they have. For a result cut back to `k`
raw neighbors, use [`expand!`](@ref). An index without members yields `res` as it is.
"""
expand(G::AbstractSearchIndex, q, res::AbstractMetricQueue) = Expanded(G, q, IdDistView(res))
expand(G::AbstractSearchIndex, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) = Expanded(G, q, _iditems(ids, dists))

_iditems(ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) =
    (IdDist(ids[i], dists[i]) for i in eachindex(ids) if ids[i] != 0)

struct Expanded{G<:AbstractSearchIndex,Q,I}
    index::G
    q::Q
    items::I
end

Base.IteratorSize(::Type{<:Expanded}) = Base.SizeUnknown()
Base.eltype(::Type{<:Expanded}) = IdDist

function Base.iterate(e::Expanded, state=nothing)
    if state === nothing
        r = iterate(e.items)
        r === nothing && return nothing
        p, s = r
        return p, (s, p, 1)
    end
    s, p, j = state
    M = members(e.index, p.id)
    if j <= length(M)
        m = M[j]
        return IdDist(m, Float32(evaluate(distance(e.index), e.q, database(e.index)[m]))), (s, p, j + 1)
    end
    r = iterate(e.items, s)
    r === nothing && return nothing
    p2, s2 = r
    p2, (s2, p2, 1)
end

"""
    expand!(G, q, res::AbstractMetricQueue) -> res
    expand!(G, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) -> (ids, dists)
    expand!(G, Q::AbstractDatabase, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}) -> (knns, dists)

In place: pushes the members of every cluster in the result into it, each with its distance
to the query evaluated by the index's distance, and lets the result's own rule trim -- a
k-nn queue keeps its `k` nearest, a radius queue what falls within its radius, a column of
[`searchbatch`](@ref SimilaritySearch.searchbatch)'s matrices its `k` rows. The first stage, [`search`](@ref), answered
with one representative per cluster; after this the result holds raw neighbors, duplicates
included, which is what [`macrorecall`](@ref) against an exhaustive gold expects. The matrix
form runs the columns in parallel. An index without members returns the result untouched.
"""
function expand!(G::AbstractSearchIndex, q, res::AbstractMetricQueue)
    _expand!(G, q, res)
    res
end

"Expands in place and returns how many member distances it evaluated."
function _expand!(G::AbstractSearchIndex, q, res::AbstractMetricQueue)::Int
    reps = UInt32[]
    for id in IdView(res)
        isempty(members(G, id)) || push!(reps, id)
    end
    isempty(reps) && return 0
    dist = distance(G); db = database(G)
    n = 0
    for r in reps, m in members(G, r)
        push_item!(res, m, Float32(evaluate(dist, q, db[m])))
        n += 1
    end
    n
end

function expand!(G::AbstractSearchIndex, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32})
    k = length(ids)
    items = collect(_iditems(ids, dists))
    res = knnqueue(KnnSorted, ids, dists)
    reuse!(res)
    for p in items
        push_item!(res, p.id, p.dist)
    end
    expand!(G, q, res)
    for j in length(res)+1:k           # the slots the result does not fill are empty, as searchbatch leaves them
        ids[j] = 0
        dists[j] = typemax(Float32)
    end
    ids, dists
end

function expand!(G::AbstractSearchIndex, Q::AbstractDatabase, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32})
    size(knns, 2) == length(Q) || throw(DimensionMismatch("expand!: $(size(knns, 2)) result columns for $(length(Q)) queries"))
    @BATCHES getminbatch(length(Q)) for i in 1:length(Q)
        expand!(G, Q[i], view(knns, :, i), view(dists, :, i))
    end
    knns, dists
end

"""
    setmember!(G::SearchGraph, id, rep)

Makes `id` a member of `rep`: registers it and leaves it with the single edge to `rep`.
The caller must have kept `rep` out of `id`'s reverse links.
"""
function setmember!(G::SearchGraph, id::Integer, rep::Integer)
    addmember!(G.members, rep, id)
    _setneighbors!(G.adj, id, UInt32(rep))
    G
end

function _setneighbors!(adj::AdjList, id::Integer, rep::UInt32)
    lock(adj.glock) do
        id > length(adj) && resize!(adj, id)
        v = adj.end_point[id]
        empty!(v)
        push!(v, rep)
    end
end

"""
    isnearduplicate(ctx::SearchGraphContext, neighborhood::AbstractKnnQueue) -> Bool

Whether [`find_neighborhood!`](@ref) found the item to be a near duplicate: then the
neighborhood holds exactly one entry, the twin it must become a member of (unresolved: the
twin may itself be settled as a member in the same block; see [`resolvemembers!`](@ref)).
"""
isnearduplicate(ctx::SearchGraphContext, neighborhood::AbstractKnnQueue) =
    length(neighborhood) == 1 && maximum(neighborhood) <= ctx.neighborhood.neardup

"""
    resolvemembers!(G::SearchGraph, sp, ep, twin) -> Vector{Tuple{UInt32,UInt32}}

Settles the near duplicates of one insertion block `sp:ep`, where `twin[i]` is the twin
[`find_neighborhood!`](@ref) found for object `sp + i - 1`, or `0` for a node. Twins inside
the block are joined by union-find, so a cluster whose objects arrived in the same block gets
one representative: when any of its objects is the twin of an object outside the block, that
object's representative; otherwise the component's smallest *node* -- an object that
computed a neighborhood of its own (`twin == 0`), which a component always has, since every
twin points at an earlier object. A twin never represents: it has no neighborhood to stand
on, which is what `rebuild` would otherwise hand a cluster whose old representative carries
a larger id than its members. Returns the `(member,
representative)` pairs and **empties the adjacency of every member** -- a block object that
had computed a neighborhood of its own and turns out to be a member loses it -- so that the
reverse links connected afterwards skip them; the caller registers the pairs with
[`setmember!`](@ref) after that.
"""
function resolvemembers!(G::SearchGraph, sp::Integer, ep::Integer, twin::AbstractVector{UInt32})
    n = ep - sp + 1
    parent = collect(UInt32(1):UInt32(n))
    function find(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        i
    end
    outside = zeros(UInt32, n)
    @inbounds for i in 1:n
        t = twin[i]
        t == 0 && continue
        if sp <= t <= ep
            a, b = find(i), find(t - sp + 1)
            a != b && (parent[max(a, b)] = min(a, b))      # the root is the component's smallest id
        else
            outside[i] = representative(G, t)
        end
    end
    comprep = Dict{UInt32,UInt32}()        # a component's representative outside the block, if any
    compnode = Dict{UInt32,UInt32}()       # else its smallest node
    @inbounds for i in 1:n
        r = find(i)
        if outside[i] != 0
            comprep[r] = min(get(comprep, r, typemax(UInt32)), outside[i])
        elseif twin[i] == 0
            compnode[r] = min(get(compnode, r, typemax(UInt32)), UInt32(sp + i - 1))
        end
    end
    settled = Tuple{UInt32,UInt32}[]
    repof = Dict{UInt32,UInt32}()
    @inbounds for i in 1:n
        r = find(i)
        id = UInt32(sp + i - 1)
        rep = get(comprep, r) do
            get(compnode, r, UInt32(sp + r - 1))
        end
        rep == id && continue
        push!(settled, (id, rep))
        repof[id] = rep
        _setneighbors!(G.adj, id, UInt32[])   # emptied until the reverse links are done
    end
    # the block's nodes chose their neighbors among the block's earlier objects before these
    # were settled; a neighbor that became a member is replaced by its representative, which
    # stands at the same place, so that nothing links to a member
    isempty(repof) || @inbounds for i in 1:n
        id = UInt32(sp + i - 1)
        haskey(repof, id) && continue
        _replaceneighbors!(G.adj, id, repof)
    end
    settled
end

function _replaceneighbors!(adj::AdjList, id::Integer, repof::Dict{UInt32,UInt32})
    lock(adj.glock) do
        v = adj.end_point[id]
        any(x -> haskey(repof, x), v) || return
        for j in eachindex(v)
            v[j] = get(repof, v[j], v[j])
        end
        unique!(v)
        filter!(!=(UInt32(id)), v)
    end
end

_setneighbors!(adj::AdjList, id::Integer, ::Vector{UInt32}) = lock(adj.glock) do
    id > length(adj) && resize!(adj, id)
    empty!(adj.end_point[id])
end
