# This file is a part of SimilaritySearch.jl
#
"""
    approx_by_hints!(index::SearchGraph, ctx, q, hints, res, vstate)

Approximate the result using a set of hints (the set of identifiers (integers)) behints  `hints`
"""
function approx_by_hints!(index::SearchGraph, ctx, q, hints::T, res, vstate) where {T<:Union{AbstractVector,Tuple,Integer,Set}}
    for objID in hints
        enqueue_item!(index, ctx, q, database(index, objID), res, objID, vstate)
    end

    res
end

"""
    AdjacentStoredHints{DB<:AbstractDatabase}(hints::DB, map::Vector{Int32})

Stores a materialized copy of the hint objects (`hints`) together with the identifiers
(`map`) of the corresponding elements in the original dataset. This allows hint objects to
be kept in an alternative database representation `DB` (e.g., a `MatrixDatabase`) instead of
being fetched by identifier from the main dataset on every access; see [`matrixhints`](@ref).

# Fields
- `hints`: database holding the materialized hint objects
- `map`: identifiers, in the original dataset, of each corresponding hint object
"""
struct AdjacentStoredHints{DB<:AbstractDatabase}
    hints::DB
    map::Vector{Int32}
end

Base.length(A::AdjacentStoredHints) = length(A.hints)

"""
    matrixhints(index::SearchGraph, ::Type{DBType}=MatrixDatabase) where {DBType<:AbstractDatabase}

Materializes the objects currently referenced by `index`'s hints (stored as a list of
identifiers) into an [`AdjacentStoredHints`](@ref) object backed by `DBType`, which can
improve cache locality when hints are repeatedly accessed while searching. Returns a copy of
`index` with the new hints installed (`index` itself is not modified).

# Arguments
- `index`: the search graph whose current hints will be materialized
- `DBType`: the database type used to store the materialized hint objects, defaults to `MatrixDatabase`

# Examples

```julia
G = SearchGraph(; dist, db)
index!(G, ctx)
G = matrixhints(G)  # hints are now stored using a MatrixDatabase
```
"""
function matrixhints(index::SearchGraph, ::Type{DBType}=MatrixDatabase) where {DBType<:AbstractDatabase}
    h = Vector{Int32}(index.hints)
    s = SubDatabase(database(index), h)
    @set index.hints = AdjacentStoredHints(DBType(s), h)
end

function approx_by_hints!(index::SearchGraph, ctx, q, h::AdjacentStoredHints, res, vstate)
    for (i, objID) in enumerate(h.map)
        enqueue_item!(index, ctx, q, h.hints[i], res, objID, vstate)
    end

    res
end

"""
    RandomHints(; logbase=1.1)

A [`Callback`](@ref) that selects search hints as a random sample of the dataset. Sampled
objects are only accepted as hints if they (and their neighborhood) are not already
part of the neighborhood of a previously accepted hint and have a minimum degree, which
tends to favor well-connected entry points for searches.

# Keyword Arguments
- `logbase`: log base used to compute the number of hints to keep, i.e., approximately
  `log(logbase, n)` hints are kept for a dataset of `n` elements.

# Examples

```julia
ctx = SearchGraphContext(; hints_callback=RandomHints(; logbase=1.2))
```
"""
@kwdef mutable struct RandomHints <: Callback
    logbase::Float32 = 1.1
end

"""
    execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::RandomHints)

`SearchGraph`'s callback for selecting hints at random, see [`RandomHints`](@ref).
"""
function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::RandomHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    V = Set{Int}()

    for _ in 1:n
        objID = rand(1:n)
        objID in V && continue
        if !(objID in V)
            N = neighbors(index.adj, objID)
            length(N) <= 2 && continue
            push!(V, objID)
            union!(V, N)
            for child in N
                child ∈ V || union!(V, neighbors(index.adj, child))
            end
            push!(index.hints, objID)
        end

        length(index.hints) >= m && break
    end

    #@info "HINTS-size:" length(index.hints )
end

#=function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::RandomHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    V = Set{Int}()

    for objID in 1:n
        #objID = rand(1:n)
        if !(objID in V)
            N = neighbors(index.adj, objID)
            length(N) <= 3 && continue
            push!(V, objID)
            union!(V, N)
            push!(index.hints, objID)
        end

        #length(index.hints) >= m && break
    end

    @info "HINTS-size:" length(index.hints )
end=#

"""
    DisjointHints(; logbase=1.1)

A [`Callback`](@ref) that selects search hints as a small subsample of mutually disjoint
objects, i.e., objects whose neighborhoods do not overlap with the neighborhoods of other
selected hints. Candidates are visited in decreasing order of how much their degree
deviates from the mean degree of the graph.

# Keyword Arguments
- `logbase`: log base used to compute the number of hints to keep, i.e., approximately
  `log(logbase, n)` hints are kept for a dataset of `n` elements.

# Examples

```julia
ctx = SearchGraphContext(; hints_callback=DisjointHints(; logbase=1.2))
```
"""
@kwdef mutable struct DisjointHints <: Callback
    logbase::Float32 = 1.1
end

"""
    execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::DisjointHints)

`SearchGraph`'s callback for selecting disjoint hints, see [`DisjointHints`](@ref).
"""
function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::DisjointHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    meansize = mean(length(neighbors(index.adj, i)) for i in 1:n)
    res = knnqueue(ctx, m)
    for i in 1:n
        push_item!(res, i, abs(length(neighbors(index.adj, i)) - meansize))
    end

    V = Set{Int}()
    for item in res
        i = item.id
        i in V && continue
        push!(index.hints, i)
        push!(V, i)
        union!(V, neighbors(index.adj, i))
    end
end

"""
    KDisjointHints(; logbase=1.1, disjoint=3, expansion=4)

A [`Callback`](@ref) that selects search hints by randomly visiting candidate objects and
greedily accepting them as hints while marking their expanded neighborhood (up to
`expansion` hops away) as visited, so that accepted hints tend to have disjoint
neighborhoods.

# Keyword Arguments
- `logbase`: log base used to compute the number of hints to keep, i.e., approximately
  `log(logbase, n)` hints are kept for a dataset of `n` elements.
- `disjoint`: parameter reserved to control the degree of disjointness enforced among hints
  (not read by the current sampling procedure).
- `expansion`: number of hops used to expand the neighborhood of an accepted hint before
  marking it as visited (i.e., excluded from being selected again).

# Examples

```julia
ctx = SearchGraphContext(; hints_callback=KDisjointHints(; logbase=1.2, expansion=3))
```
"""
@kwdef struct KDisjointHints <: Callback
    logbase::Float32 = 1.1
    disjoint::Int32 = 3
    expansion::Int32 = 4
end

"""
    execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::KDisjointHints)

`SearchGraph`'s callback for selecting disjoint hints, see [`KDisjointHints`](@ref).
"""
function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::KDisjointHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    sample = unique(rand(Int32(1):Int32(n), opt.expansion * m))
    m = min(length(sample), m)
    sort!(sample, by=i -> length(neighbors(index.adj, i)), rev=true)
    IType = eltype(index.hints)
    visited = Set{IType}()
    empty!(index.hints)
    E = Pair{IType,Int32}[]
    i = 1
    while length(index.hints) < m && i < length(sample)
        # p = rand(1:n)
        p = sample[i]
        i += 1
        p in visited && continue
        push!(index.hints, p)
        push!(visited, p)
        # visit the neighborhood with some expansion factor
        push!(E, p => 0)
        while length(E) > 0
            parent, e = pop!(E)
            for child in keys(neighbors(index.adj, parent))
                if !(child in visited)
                    push!(visited, child)
                    e + 1 <= opt.expansion && push!(E, child => e + 1)
                end
            end
        end
    end
end

"""
    EpsilonHints(; quantile=0.01, epsilon=0.0f0, minepsilon=1e-5, samplesize=sqrt, maxsize=x->log(1.1,x))

A [`Callback`](@ref) that selects search hints as a random sample of the dataset from which
near-duplicate objects (those closer than a distance threshold `epsilon`) have been removed,
so that the resulting hints are spread out over the dataset.

# Keyword Arguments
- `quantile`: if greater than `0`, `epsilon` is instead estimated as this quantile of a
  sample of pairwise distances; use `quantile<=0` to use the fixed `epsilon` value instead.
- `epsilon`: fixed near-duplicate distance threshold, used only when `quantile<=0`.
- `minepsilon`: lower bound enforced on the estimated `epsilon` when `quantile>0`.
- `samplesize`: function of the dataset size `n` used to determine how many objects are
  initially sampled before near-duplicate removal.
- `maxsize`: function of the dataset size `n` used to determine the maximum number of hints
  to keep (extra hints beyond this size are discarded at random).

# Examples

```julia
ctx = SearchGraphContext(; hints_callback=EpsilonHints(; quantile=0.05))
```
"""
mutable struct EpsilonHints <: Callback
    epsilon::Float32
    minepsilon::Float32
    quantile::Float32
    samplesize::Function
    maxsize::Function
end

EpsilonHints(; quantile=0.01, epsilon=0.0f0, minepsilon=1e-5, samplesize=sqrt, maxsize=x -> log(1.1, x)) =
    EpsilonHints(convert(Float32, epsilon),
        convert(Float32, minepsilon),
        convert(Float32, quantile),
        samplesize,
        maxsize)

"""
    execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::EpsilonHints)

`SearchGraph`'s callback for selecting near-duplicate-free hints, see [`EpsilonHints`](@ref).
"""
function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::EpsilonHints)
    n = length(index)
    m = min(n, ceil(Int, opt.samplesize(n)))
    s = rand(1:n, m) |> unique! |> sort!

    sample = VectorDatabase(s)
    out = VectorDatabase(Int32[])
    dist = DistanceWithIdentifiers(distance(index), database(index))
    E = ExhaustiveSearch(; dist, db=out)
    ϵ = opt.quantile <= 0.0 ? opt.epsilon : let
        D = distsample(dist, sample; samplesize=m)
        max(opt.minepsilon, quantile(D, opt.quantile))
    end

    neardup(E, GenericContext(), sample, ϵ)
    v = out.vecs # internals of VectorDatabase
    max_ = ceil(Int, opt.maxsize(n))
    if length(v) > max_
        shuffle!(v)
        resize!(v, max_)
    end

    resize!(index.hints, length(v))
    index.hints .= v
end

"""
    KCentersHints(; logbase=1.1, powsample=1.5, qdiscard=0.1)

A [`Callback`](@ref) that selects search hints using a k-centers (farthest-first traversal)
strategy. A random sample of candidate objects is drawn from the dataset (filtered to
exclude atypically low- or high-degree vertices), and a set of `k` centers is computed over
that sample using a farthest-first traversal. Centers that end up receiving too few nearest
neighbor assignments (i.e., that look redundant or of little use as entry points) are
discarded before the remaining ones are used as hints.

# Keyword Arguments
- `logbase`: log base used to compute the number of centers/hints to search for, i.e.,
  approximately `log(logbase, n) + 1` centers are computed for a dataset of `n` elements.
- `powsample`: exponent used to determine the size of the candidate sample from which
  centers are computed, i.e., `k^powsample` candidates are sampled (`k` being the number of
  centers).
- `qdiscard`: quantile, over the number of nearest-neighbor assignments received by each
  center, used to discard the least used centers (centers below this quantile are dropped).

# Examples

```julia
ctx = SearchGraphContext(; hints_callback=KCentersHints(; logbase=1.2))
```
"""
mutable struct KCentersHints <: Callback
    logbase::Float32
    powsample::Float32
    qdiscard::Float32
end

KCentersHints(; logbase=1.1, powsample=1.5, qdiscard=0.1) = KCentersHints(logbase, powsample, qdiscard)

"""
    execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::KCentersHints)

`SearchGraph`'s callback for selecting k-centers-based hints, see [`KCentersHints`](@ref).
"""
function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::KCentersHints)
    n = length(index)
    k = min(n ÷ 2, ceil(Int, log(opt.logbase, n))) + 1
    @assert n > k
    m = min(n, ceil(Int, k^opt.powsample))
    #m / n
    D = let s = rand(1:n, m) |> unique! #|> sort!
        degrees = neighbors_length.(Ref(index.adj), s)
        min_, max_ = quantile(degrees, [0.25, 0.95])
        s = [j for (i, j) in enumerate(s) if min_ <= degrees[i] <= max_]
        sort!(s)
        SubDatabase(database(index), s)
    end
    
    A = fft(distance(index), D, k; ctx.verbose)
    M = Dict(c => i for (i, c) in enumerate(A.centers))
    #@show M
    #@show A.nn
    # @show A unique(A.nn) D.map
    count = zeros(Int, length(M))
    for nn in A.nn
        count[M[nn]] += 1
    end
    x = quantile(count, opt.qdiscard)
    C = A.centers[count.>=x]

    verbose(ctx) && @info "KCentersHints: n=$n, m=$m, k=$k, numcenters=$(length(A.centers)), C=$(length(C))"
    resize!(index.hints, length(C))
    index.hints .= D.map[C]
end

