# This file is a part of SimilaritySearch.jl

"""
    BKT(dist::Dist.Metric, db::AbstractDatabase; checkmetric::Bool=true)
    BKT(db::AbstractDatabase; dist::Dist.Metric, checkmetric::Bool=true)

A [BK-tree](https://en.wikipedia.org/wiki/BK-tree): an **exact** index for metrics whose
distance takes few distinct, *integer* values -- e.g. [`Dist.Seqs.Levenshtein`](@ref),
[`Dist.Bits.Hamming`](@ref). Each internal node holds a pivot object and buckets its
subtree by the *exact* integer distance to that pivot, so a query prunes a whole subtree
with a single distance evaluation.

The structure is built once with [`index!`](@ref); it does not support incremental
insertion (like [`Sat`](@ref), and unlike [`SearchGraph`](@ref)).

# Why integer-valued distances only

The pruning rule is a direct consequence of the triangle inequality: every object `x`
hanging under the child keyed `k` of a node `p` satisfies `d(p, x) == k` *exactly*, hence
`d(q, x) >= |d(q, p) - k|`, and the whole subtree is discarded when that bound exceeds the
covering radius of the result. A continuous distance breaks this twice over: the exact-key
bucketing degenerates (almost every pair gets its own bucket, so the tree becomes a list
with no pruning power), and the rounding that assigns a bucket breaks the `d(p, x) == k`
invariant the bound relies on.

**This is a contract, not a check.** `index!` rounds every distance it observes to an
`Int32` key and trusts the caller: handing `BKT` a continuous distance builds a tree that
answers queries without complaining and can quietly miss true neighbors. Verifying it
cheaply is not possible anyway -- the build only ever sees a fraction of the `n^2` pairs --
so it is stated here rather than half-enforced.

# Why a `Metric` and not a `SemiMetric`

Pruning needs the triangle inequality, which `Dist.SemiMetric` does not promise. A relevant
example lives in this very package: [`Dist.Seqs.DamerauLevenshtein`](@ref) is integer-valued
but deliberately typed `SemiMetric` because the restricted/OSA variant violates the triangle
inequality, so a `BKT` built on it can miss true neighbors. The constructor rejects a
non-`Metric` distance for this reason. `checkmetric=false` overrides it, for either of two
quite different reasons: the distance really is a metric and is merely typed loosely, *or*
you knowingly accept an approximate index (see below).

# Searching by Damerau-Levenshtein

Two routes, both useful, neither free:

1. **Key the tree by `DamerauLevenshtein` itself** (`checkmetric=false`). Pruning is no
   longer provably sound, so this is an *approximate* index. Measured on a 20k-word
   dictionary with 200 typo queries (transpositions included), it lost nothing at all --
   recall `1.0` at radius 1, 2 and 3 -- at 7.9%/39%/68% of an exhaustive scan. Empirical
   evidence on one corpus, not a guarantee: it says OSA's triangle-inequality violations are
   too rare or too small to bite at these thresholds, not that they cannot.
2. **Key the tree by [`Dist.Seqs.Levenshtein`](@ref), search at radius `2r`, then filter the
   candidates by `DamerauLevenshtein`.** Exact, with no reliance on OSA being a metric: OSA
   allows every Levenshtein operation plus adjacent transposition, and each transposition can
   be replayed as two substitutions, so `DL <= Lev <= 2*DL` and `DL(q,x) <= r` implies
   `Lev(q,x) <= 2r`. The doubled radius costs pruning: on that same dictionary it took
   33%/79%/107% of an exhaustive scan for `r = 1/2/3`, i.e. by `r = 3` it is already worse
   than not having an index.

Both degrade quickly as the threshold grows, which is the general caveat below in its
sharpest form: a BK-tree over edit distance is a *small-threshold* structure.

# When a BK-tree pays off

Pruning bites when the result's covering radius is *small next to the spread of the distance
distribution* -- the shape of dictionary lookup/spelling correction, where the neighbors sought
sit 1-2 edits away while the bulk of the collection sits much farther. It does nothing for a
query whose radius covers most of that spread (a large `k` over data with no near neighbors,
e.g. uniformly random strings): every child key then falls inside `[d(q,p)-r, d(q,p)+r]`, the
whole tree is visited, and the search degenerates into an exhaustive scan -- correct, just
with no speedup. That is a property of the data, not of this implementation, and it is why
this index ships alongside (rather than replacing) [`ExhaustiveSearch`](@ref).

# Arguments
- `dist`: the distance function; must be integer-valued (see above)
- `db`: the database to index

# Keyword Arguments
- `checkmetric`: whether a non-`Dist.Metric` distance is rejected (default `true`)

# Examples

```julia
using SimilaritySearch

db = VectorDatabase([collect(w) for w in ["form", "from", "fort", "fore", "ford"]])
bkt = BKT(Dist.Seqs.Levenshtein(), db)
ctx = GenericContext()
index!(bkt, ctx)
search(bkt, ctx, collect("fond"), knnqueue(KnnSorted, 3))
```
"""
struct BKT{DistanceType<:Dist.PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType

    # The tree. Node ids are database ids: node `i` holds the object `db[i]`. A node is
    # either internal (it has children, each keyed by the exact distance from `db[i]` to
    # every object of that child's subtree) or a long leaf (it carries `bucketlen` further
    # objects, stored contiguously in `bucket`). A database id that is neither -- an object
    # sitting inside some leaf's bucket -- simply keeps its slots at 0. `index!` fills all
    # of this; the vectors are empty until then.
    root::Base.RefValue{UInt32}   # 0 while the tree is empty
    childstart::Vector{Int32}     # per node: 1-based offset into childkey/childnode; 0 = no children
    childcount::Vector{Int32}     # per node: number of children
    childkey::Vector{Int32}       # per edge: d(pivot, x) for every x in that child's subtree,
    childnode::Vector{UInt32}     # per edge: the child node -- both ascending by key within a node
    bucketstart::Vector{Int32}    # per node: 1-based offset into bucket; 0 = no bucket
    bucketlen::Vector{Int32}      # per node: number of objects in its bucket
    bucket::Vector{UInt32}        # flat storage of every long leaf's objects
end

function BKT(dist::Dist.PreMetric, db::AbstractDatabase; checkmetric::Bool=true)
    if checkmetric && !(dist isa Dist.Metric)
        throw(ArgumentError("""
            BKT prunes with the triangle inequality, which $(typeof(dist)) does not promise \
            (it is not a `Dist.Metric`); a BKT built on it can miss true neighbors. \
            Pass `checkmetric=false` if this distance really is a metric but is not typed as one."""))
    end

    BKT(dist, db, Ref(zero(UInt32)),
        Int32[], Int32[], Int32[], UInt32[], Int32[], Int32[], UInt32[])
end

BKT(db::AbstractDatabase; dist::Dist.PreMetric, checkmetric::Bool=true) = BKT(dist, db; checkmetric)

@inline database(bkt::BKT) = bkt.db
@inline database(bkt::BKT, i) = bkt.db[i]
@inline distance(bkt::BKT) = bkt.dist
@inline Base.length(bkt::BKT) = length(bkt.db)

"""
    getcontext(bkt::BKT; kwargs...)

A [`GenericContext`](@ref) -- a `BKT` search allocates no scratch of its own, so it needs no
dedicated context type. Not part of the formal `AbstractSearchIndex` interface, a convenience.
"""
getcontext(::BKT; kwargs...) = GenericContext(; kwargs...)

"""
    _ndistinct!(scratch, v, lo, hi)

Number of distinct values in `v[lo:hi]`, using `scratch` as workspace (no allocation).
"""
function _ndistinct!(scratch::Vector{Int32}, v::Vector{Int32}, lo::Integer, hi::Integer)
    @inbounds for i in lo:hi
        scratch[i] = v[i]
    end

    sort!(view(scratch, lo:hi))
    n = 1
    @inbounds for i in (lo+1):hi
        scratch[i] != scratch[i-1] && (n += 1)
    end

    n
end

"""
    _firstkeyge(bkt::BKT, lo::Integer, hi::Integer, d)

Index of the first edge in `lo:hi` whose key is `>= d`, or `hi + 1` when there is none
(`bkt.childkey[lo:hi]` is ascending). Plain binary search, kept local to avoid depending on
`Base`'s internal range-restricted `searchsortedfirst`.
"""
@inline function _firstkeyge(bkt::BKT, lo::Integer, hi::Integer, d)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        if bkt.childkey[mid] < d
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    lo
end

"""
    index!(bkt::BKT, ctx::AbstractContext; npivots=2, minleaf=12, rng=Random.default_rng())

Builds the tree over the whole `database(bkt)`, top-down: at each node it picks a pivot,
partitions the remaining objects by their exact integer distance to it, and recurses into
each resulting bucket. Returns `bkt`. The tree must be empty (`BKT` is build-once, it has no
incremental insertion).

# Keyword Arguments
- `npivots`: how many pivot candidates are sampled per node; the one producing the **most
  distinct distance values** wins, i.e. the one splitting its objects into the most buckets.
  The winner's distances are reused to partition, so the build costs one extra pass per
  *rejected* candidate: `npivots` is a straight multiplier on build time. Keep it small --
  `1` disables the choice altogether, and on a 30k-word dictionary going from `1` to `2`
  improved every query shape measured (k-NN and range alike) by 10-20%, while `3` and `5`
  bought nothing beyond it and were often worse: the criterion also rewards *outlier*
  pivots, which see many distinct distances precisely because they sit far from everything,
  and the more candidates are drawn the likelier one is picked.
- `minleaf`: objects per **long leaf**. A group this size or smaller becomes a leaf holding a
  plain list, scanned exhaustively, instead of a sub-tree. This trades query cost for build
  cost and size, and it is a real trade in both directions: on that same dictionary, going
  from `4` to `32` shrank the tree ~7x (4622 to 613 internal nodes) and the build ~20%, and
  cost ~40% more distance evaluations per query. Pass `1` to build the tree all the way down.
- `rng`: random source used to sample pivot candidates.
"""
function index!(bkt::BKT, ctx::AbstractContext;
        npivots::Int=2, minleaf::Int=12, rng::AbstractRNG=Random.default_rng())
    npivots >= 1 || throw(ArgumentError("npivots must be >= 1, got $npivots"))
    minleaf >= 1 || throw(ArgumentError("minleaf must be >= 1, got $minleaf"))
    bkt.root[] == 0 && isempty(bkt.bucket) ||
        throw(ArgumentError("index! needs an empty BKT: it is a build-once index, it cannot grow or be rebuilt in place"))

    n = length(bkt.db)
    n == 0 && return bkt

    for v in (bkt.childstart, bkt.childcount, bkt.bucketstart, bkt.bucketlen)
        resize!(v, n)
        fill!(v, 0)
    end

    cost = _build!(bkt, n, npivots, minleaf, rng)

    add_distance_evaluations!(ctx, cost)
    OBSERVE(ctx, :add!, bkt, 1, n)
    @inform ctx "add! sp=1 ep=$n" index=bkt
    bkt
end

function _build!(bkt::BKT, n::Int, npivots::Int, minleaf::Int, rng::AbstractRNG)
    dist = distance(bkt)
    db = database(bkt)

    # `work` holds a permutation of the database ids; a node owns the contiguous range
    # `lo:hi` of it, and partitioning a node just reorders its own range in place.
    work = Vector{IdIntDist}(undef, n)
    @inbounds for i in 1:n
        work[i] = IdIntDist(i, 0)
    end

    best = Vector{Int32}(undef, n)     # distances of the winning candidate, reused to partition
    cand = Vector{Int32}(undef, n)     # distances of the candidate being evaluated
    scratch = Vector{Int32}(undef, n)  # workspace of _ndistinct!

    # (lo, hi, slot) -- an explicit worklist: a pending node is fully described by the range
    # of `work` it owns plus the edge slot naming it (0 for the root, which nothing points
    # to). A node's own id is only known once its pivot is chosen, which is why the parent
    # reserves the slot up front and the child fills it in.
    stack = [(Int32(1), Int32(n), Int32(0))]
    cost = 0

    @inbounds while !isempty(stack)
        lo, hi, slot = pop!(stack)
        s = hi - lo + 1

        if s <= minleaf  # long leaf: its first object represents it, the rest go to `bucket`
            node = work[lo].id
            _attach!(bkt, slot, node)
            if s > 1
                bkt.bucketstart[node] = length(bkt.bucket) + 1
                bkt.bucketlen[node] = s - 1
                for i in (lo+1):hi
                    push!(bkt.bucket, work[i].id)
                end
            end

            continue
        end

        # pivot selection: keep the candidate splitting `lo:hi` into the most buckets
        bestpos = lo
        bestnd = -1
        m = min(npivots, s)
        for t in 1:m
            # every object is a candidate when there are no more of them than candidates
            cpos = m == s ? (lo + t - 1) : rand(rng, lo:hi)
            p = work[cpos].id
            for i in lo:hi
                # the caller guarantees these are integer-valued (see BKT's docstring); a
                # fractional distance would round into a bucket it does not belong to
                cand[i] = round(Int32, Dist.evaluate(dist, db[p], db[work[i].id]))
            end
            cost += s

            nd = _ndistinct!(scratch, cand, lo, hi)
            if nd > bestnd
                bestnd = nd
                bestpos = cpos
                cand, best = best, cand  # keep the winner's distances, they partition below
            end
        end

        for i in lo:hi
            work[i] = IdIntDist(work[i].id, best[i])
        end

        work[lo], work[bestpos] = work[bestpos], work[lo]  # the pivot leads its own range
        pivot = work[lo].id
        _attach!(bkt, slot, pivot)

        sort!(view(work, (lo+1):hi), by=e -> e.dist)

        # every equal-key group becomes a child; they are appended in ascending key order,
        # which is what `search` binary-searches on
        bkt.childstart[pivot] = length(bkt.childkey) + 1
        nc = 0
        i = lo + 1
        while i <= hi
            k = work[i].dist
            j = i
            while j < hi && work[j+1].dist == k
                j += 1
            end

            push!(bkt.childkey, k)
            push!(bkt.childnode, 0)  # filled once this child's own pivot is chosen
            nc += 1
            push!(stack, (Int32(i), Int32(j), Int32(length(bkt.childkey))))
            i = j + 1
        end

        bkt.childcount[pivot] = nc
    end

    cost
end

"Names `node` as the root (`slot == 0`) or as the child occupying the edge `slot`."
@inline function _attach!(bkt::BKT, slot::Integer, node::Integer)
    @inbounds if slot == 0
        bkt.root[] = node
    else
        bkt.childnode[slot] = node
    end
end

"""
    searchtree(bkt::BKT, ctx::AbstractContext, q, p::Integer, res::AbstractMetricQueue)

Solves `q` in the subtree rooted at `p`, returning the number of distance evaluations spent.
"""
function searchtree(bkt::BKT, ctx::AbstractContext, q, p::Integer, res::AbstractMetricQueue)
    dist = distance(bkt)
    dqp = Dist.evaluate(dist, q, database(bkt, p))
    cost = 1
    push_item!(res, p, dqp)

    @inbounds begin
        # a long leaf's objects all sit at the same distance from *this node's parent*, so
        # there is nothing left to prune among them: scan them all
        if bkt.bucketlen[p] > 0
            sp = bkt.bucketstart[p]
            for i in sp:(sp+bkt.bucketlen[p]-1)
                o = bkt.bucket[i]
                push_item!(res, o, Dist.evaluate(dist, q, database(bkt, o)))
            end

            cost += bkt.bucketlen[p]
        end

        nc = bkt.childcount[p]
        if nc > 0
            # Every x under the child keyed k satisfies d(p, x) == k, hence
            # d(q, x) >= |dqp - k|: that child's whole subtree is out once |dqp - k| exceeds
            # the covering radius. Children are visited by increasing |dqp - k| -- two
            # pointers walking outwards from dqp over the ascending keys -- because the
            # radius only shrinks as candidates are found, and starting with the child most
            # likely to hold them is what makes the rest prunable. |dqp - k| grows
            # monotonically along each side, so a side that fails the test is done for good.
            cs = bkt.childstart[p]
            ce = cs + nc - 1
            j = _firstkeyge(bkt, cs, ce, dqp)  # keys >= dqp
            i = j - 1                          # keys < dqp

            while true
                r = covradius(res)
                dl = i >= cs ? dqp - bkt.childkey[i] : typemax(Float32)
                dr = j <= ce ? bkt.childkey[j] - dqp : typemax(Float32)
                dl > r && (i = cs - 1; dl = typemax(Float32))
                dr > r && (j = ce + 1; dr = typemax(Float32))
                dl == typemax(Float32) && dr == typemax(Float32) && break

                if dl <= dr
                    cost += searchtree(bkt, ctx, q, bkt.childnode[i], res)
                    i -= 1
                else
                    cost += searchtree(bkt, ctx, q, bkt.childnode[j], res)
                    j += 1
                end
            end
        end
    end

    cost
end

"""
    search(bkt::BKT, ctx::AbstractContext, q, res::AbstractMetricQueue) -> res

Solves query `q`, pushing candidates into `res`. The result is **exact**: every object the
search discards is separated from `q` by more than the covering radius of `res`, by the
triangle inequality (see [`BKT`](@ref)).

Works with a `k`-nearest-neighbor queue (the radius shrinks as `res` fills up) and with a
[`RadiusSorted`](@ref)/[`RadiusHeap`](@ref) range queue (a fixed radius) alike, since both
answer [`covradius`](@ref).
"""
function search(bkt::BKT, ctx::AbstractContext, q, res::AbstractMetricQueue)
    root = bkt.root[]
    root == 0 && return res
    cost = searchtree(bkt, ctx, q, root, res)
    add_distance_evaluations!(ctx, cost)
    res
end
