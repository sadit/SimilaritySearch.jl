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
    index!(bkt::BKT, ctx::AbstractContext; npivots=2, nsample=32, minleaf=12)

Builds the tree over the whole `database(bkt)`, top-down: at each node it picks a pivot,
partitions the remaining objects by their exact integer distance to it, and recurses into
each resulting bucket. Returns `bkt`. The tree must be empty (`BKT` is build-once, it has no
incremental insertion).

# Keyword Arguments
- `npivots`: how many pivot candidates are considered per node; the one producing the **most
  distinct distance values** wins, i.e. the one splitting its objects into the most buckets.
  `1` disables the choice altogether. On a 30k-word dictionary going from `1` to `2` improved
  every query shape measured (k-NN and range alike) by 10-20%, while `3` and `5` bought
  nothing beyond it and were often worse: the criterion also rewards *outlier* pivots, which
  see many distinct distances precisely because they sit far from everything, and the more
  candidates are drawn the likelier one is picked.
- `nsample`: how many of a node's own objects each candidate is judged against. Selection is
  a heuristic, so it is judged on a sample rather than on everything the node covers -- that
  keeps its cost `npivots * nsample` per node instead of `npivots` times the node's size,
  which would make `npivots` a straight multiplier on the whole build. Every candidate of a
  node faces the *same* sample, so they compete on equal terms.
- `minleaf`: objects per **long leaf**. A group this size or smaller becomes a leaf holding a
  plain list, scanned exhaustively, instead of a sub-tree. This trades query cost for build
  cost and size, and it is a real trade in both directions: on that same dictionary, going
  from `4` to `32` shrank the tree ~7x (4622 to 613 internal nodes) and the build ~20%, and
  cost ~40% more distance evaluations per query. Pass `1` to build the tree all the way down.

Nodes of the same level are expanded in parallel (`@BATCHES`, `scheduler=ctx.scheduler`):
they own disjoint ranges of the working permutation, so they never contend. Only the
reservation of their slots in the shared child/bucket arrays is serial, and that is a prefix
sum over the level's nodes -- `O(#nodes)`, against the `O(npivots * n)` distance evaluations
each level spends. Pivot candidates are drawn from the task-local RNG, so `Random.seed!`
still controls the build, but the tree it produces depends on the thread count."""
function index!(bkt::BKT, ctx::AbstractContext;
        npivots::Int=2, nsample::Int=32, minleaf::Int=12)
    npivots >= 1 || throw(ArgumentError("npivots must be >= 1, got $npivots"))
    minleaf >= 1 || throw(ArgumentError("minleaf must be >= 1, got $minleaf"))
    nsample >= 1 || throw(ArgumentError("nsample must be >= 1, got $nsample"))
    bkt.root[] == 0 && isempty(bkt.bucket) ||
        throw(ArgumentError("index! needs an empty BKT: it is a build-once index, it cannot grow or be rebuilt in place"))

    n = length(bkt.db)
    n == 0 && return bkt

    for v in (bkt.childstart, bkt.childcount, bkt.bucketstart, bkt.bucketlen)
        resize!(v, n)
        fill!(v, 0)
    end

    # both are bounded by n (one edge per non-root node; one bucket slot per object that is
    # not its leaf's representative), and the build grows them level by level -- reserving
    # up front turns those growths into no-ops instead of realloc-and-copy
    sizehint!(bkt.childkey, n)
    sizehint!(bkt.childnode, n)
    sizehint!(bkt.bucket, n)

    cost = _build!(bkt, ctx, n, npivots, minleaf, nsample)

    add_distance_evaluations!(ctx, cost)
    OBSERVE(ctx, :add!, bkt, 1, n)
    @inform ctx "add! sp=1 ep=$n" index=bkt
    bkt
end

"""
    IdKey(id, key)

A database id paired with its integer distance to the pivot being considered, used only
while building. Private to this module on purpose: bucket keys are integers *inside* the
tree, but everything this index hands back -- via `push_item!` into the caller's result
queue -- is the package's standard `IdDist`, i.e. `Float32` distances.
"""
struct IdKey
    id::UInt32
    key::Int32
end

"""
    _choosepivot!(bkt, dist, work, lo, hi, npivots, nsample, spos, vals, scratch) -> (bestpos, cost)

Position of the pivot chosen for the node owning `work[lo:hi]`: of `npivots` candidates, the
one whose distances to a **sample** of the node's own objects take the most distinct values,
i.e. the one that splits them into the most buckets.

The sample is what keeps selection cheap: judging a candidate against every object it covers
would make the whole build `npivots` times more expensive, while the ranking it produces is a
heuristic either way. Every candidate is judged against the *same* sample, so they compete on
equal terms. Reads `work[lo:hi]` and nothing else, so nodes of a level run this concurrently.
"""
function _choosepivot!(bkt::BKT, dist, work::Vector{IdKey}, lo::Int32, hi::Int32, npivots::Int,
        nsample::Int, spos::Vector{Int32}, vals::Vector{Int32}, scratch::Vector{Int32})
    db = database(bkt)
    s = hi - lo + 1
    ns = min(nsample, s)

    # grow-only scratch owned by the batch, so a level's thousands of nodes share three
    # buffers instead of allocating three apiece
    length(spos) < ns && (resize!(spos, ns); resize!(vals, ns); resize!(scratch, ns))
    @inbounds if ns == s
        for u in 1:ns
            spos[u] = lo + u - 1
        end
    else
        for u in 1:ns
            spos[u] = rand(lo:hi)  # TaskLocalRNG: each task draws from its own state
        end
    end

    bestpos = lo
    bestnd = -1
    cost = 0
    m = min(npivots, s)
    @inbounds for t in 1:m
        # every object is a candidate when there are no more of them than candidates
        cpos = m == s ? (lo + t - 1) : rand(lo:hi)
        p = work[cpos].id
        for u in 1:ns
            vals[u] = round(Int32, Dist.evaluate(dist, db[p], db[work[spos[u]].id]))
        end
        cost += ns

        nd = _ndistinct!(scratch, vals, 1, ns)
        if nd > bestnd
            bestnd = nd
            bestpos = cpos
        end
    end

    bestpos, cost
end

"""
    _countsort!(work, work2, cnt, lo, hi, minleaf) -> (ngroups, nbucket, nbig)

Sorts `work[(lo+1):hi]` by key **and** reports what phase D must reserve for that node: how
many equal-key groups it has, how many objects land in long-leaf buckets, and how many groups
keep branching.

A counting sort, not a comparison sort, because the keys are exactly what this whole index is
built on: small non-negative integers. It runs in `O(s + kmax)` instead of `O(s log s)`, it
allocates nothing (`work2` is a shared scratch permutation, indexed over the same disjoint
range the node owns, and `cnt` is the batch's grow-only histogram), and the histogram it
builds *is* the group structure -- so counting the groups is free rather than a second pass.
"""
function _countsort!(work::Vector{IdKey}, work2::Vector{IdKey}, cnt::Vector{Int32},
        lo::Int32, hi::Int32, minleaf::Int)
    kmax = zero(Int32)
    @inbounds for i in (lo+1):hi
        kmax = max(kmax, work[i].key)
    end

    m = kmax + 1
    length(cnt) < m && resize!(cnt, m)
    @inbounds begin
        for k in 1:m
            cnt[k] = 0
        end

        for i in (lo+1):hi
            cnt[work[i].key+1] += 1
        end

        # prefix sum into starting offsets; every nonempty key is one group, so the sizes
        # this pass already has in hand are exactly what phase D needs counted
        ngroups = nbucket = nbig = 0
        acc = lo + 1
        for k in 1:m
            c = cnt[k]
            if c > 0
                ngroups += 1
                c <= minleaf ? (nbucket += c - 1) : (nbig += 1)
            end

            cnt[k] = acc
            acc += c
        end

        for i in (lo+1):hi
            e = work[i]
            p = cnt[e.key+1]
            work2[p] = e
            cnt[e.key+1] = p + 1
        end

        for i in (lo+1):hi
            work[i] = work2[i]
        end

        return ngroups, nbucket, nbig
    end
end

"""
    _emit!(bkt, work, lo, hi, slot, coff, boff, noff, next, minleaf)

Writes a node into the tree, into slots phase B already reserved for it: names it in its
parent's edge, records its children (ascending by key, as `search` expects), copies its long
leaves' objects into `bucket`, and appends the groups that keep branching to `next`. Writes
nothing that another node of the same level also writes, so this runs concurrently too.
"""
function _emit!(bkt::BKT, work::Vector{IdKey}, lo::Int32, hi::Int32, slot::Int32,
        coff::Int32, boff::Int32, noff::Int32, next::Vector{NTuple{3,Int32}}, minleaf::Int)
    @inbounds begin
        node = work[lo].id
        _attach!(bkt, slot, node)
        s = hi - lo + 1

        if s <= minleaf  # a whole node no bigger than a long leaf: only the root can be one
            if s > 1
                bkt.bucketstart[node] = boff
                bkt.bucketlen[node] = s - 1
                for (u, i) in enumerate((lo+1):hi)
                    bkt.bucket[boff+u-1] = work[i].id
                end
            end

            return nothing
        end

        bkt.childstart[node] = coff
        e, b, x = coff, boff, noff
        i = lo + 1
        while i <= hi
            k = work[i].key
            j = i
            while j < hi && work[j+1].key == k
                j += 1
            end

            bkt.childkey[e] = k
            gsize = j - i + 1
            if gsize <= minleaf  # long leaf: its first object represents it, the rest go to `bucket`
                leaf = work[i].id
                bkt.childnode[e] = leaf
                if gsize > 1
                    bkt.bucketstart[leaf] = b
                    bkt.bucketlen[leaf] = gsize - 1
                    for u in (i+1):j
                        bkt.bucket[b] = work[u].id
                        b += 1
                    end
                end
            else
                bkt.childnode[e] = 0  # filled next level, once this child picks its pivot
                next[x] = (Int32(i), Int32(j), Int32(e))
                x += 1
            end

            e += 1
            i = j + 1
        end

        bkt.childcount[node] = e - coff
    end

    nothing
end

function _build!(bkt::BKT, ctx::AbstractContext, n::Int, npivots::Int, minleaf::Int, nsample::Int)
    dist = distance(bkt)
    db = database(bkt)

    # `work` holds a permutation of the database ids; a node owns the contiguous range
    # `lo:hi` of it, and partitioning a node just reorders its own range in place.
    work = Vector{IdKey}(undef, n)
    @inbounds for i in 1:n
        work[i] = IdKey(i, 0)
    end

    # Built breadth-first, one level at a time: every node of a level owns a disjoint range
    # of `work`, so they expand concurrently. A node is `(lo, hi, slot)` -- the range it owns
    # plus the parent edge that names it (0 for the root, which nothing points to); its own
    # id is only known once its pivot is chosen, which is why the parent reserves the slot up
    # front and the child fills it in.
    level = [(Int32(1), Int32(n), Int32(0))]
    next = NTuple{3,Int32}[]

    # Every buffer below is allocated once and resized per level, never reallocated per node:
    # `wpos`/`wpiv` peak at the first level and only shrink afterwards, and the per-node
    # vectors grow monotonically, so past the first level this loop allocates nothing.
    wpos = Vector{Int32}(undef, n)   # the level's workload: every position whose key is
    wpiv = Vector{UInt32}(undef, n)  # still unknown, paired with the pivot measuring it
    work2 = Vector{IdKey}(undef, n)  # counting-sort scratch, indexed over each node's range
    # Per-batch scratch, indexed by `@batchid()` and allocated once for the whole build
    # rather than once per parallel region (there are five of them per level). Batch ids are
    # disjoint ordinals bounded by `ctx.maxbatches` -- which is exactly what `getminbatch(ctx,
    # ...)` caps the batch count by -- so a slot is private to whichever batch holds it, the
    # same arrangement `SatContext`'s `getvstate`/`getbeam` use.
    nb = max(1, Int(ctx.maxbatches))
    bdists = [beginbatch(dist) for _ in 1:nb]
    sposb = [Int32[] for _ in 1:nb]
    valsb = [Int32[] for _ in 1:nb]
    pscrb = [Int32[] for _ in 1:nb]
    cntb = [Int32[] for _ in 1:nb]

    costs = Int[]
    woff = Int32[]
    gcount = Int32[]
    bcount = Int32[]
    ncount = Int32[]
    coff = Int32[]
    boff = Int32[]
    noff = Int32[]
    cost = 0

    while !isempty(level)
        nl = length(level)
        for v in (woff, gcount, bcount, ncount, coff, boff, noff)
            length(v) < nl && resize!(v, nl)
        end

        length(costs) < nl && resize!(costs, nl)

        # ---- phase A: pick each node's pivot from a sample of its own objects, in parallel
        minbatch = getminbatch(ctx, nl)
        @BATCHES minbatch scheduler=ctx.scheduler begin
        @BEGINBATCH
            _b = @batchid()
            bdist = bdists[_b]
            spos = sposb[_b]; vals = valsb[_b]; pscratch = pscrb[_b]
        @LOOP for t in 1:nl
            lo, hi, _ = level[t]
            if hi - lo + 1 <= minleaf   # already a long leaf: no pivot to choose
                costs[t] = 0
            else
                bestpos, c = _choosepivot!(bkt, bdist, work, lo, hi, npivots, nsample,
                                           spos, vals, pscratch)
                costs[t] = c
                work[lo], work[bestpos] = work[bestpos], work[lo]  # the pivot leads its range
            end
        end
        @END
        end

        # ---- the workload: one entry per object whose key is still unknown, tagged with the
        # pivot that will measure it. Flat and dense, so the root -- a single node covering
        # everything -- parallelizes exactly as well as a level made of thousands of nodes.
        w = 0
        @inbounds for t in 1:nl
            lo, hi, _ = level[t]
            woff[t] = w + 1
            cost += costs[t]
            hi - lo + 1 <= minleaf || (w += hi - lo)
        end

        resize!(wpos, w)
        resize!(wpiv, w)
        @BATCHES minbatch scheduler=ctx.scheduler for t in 1:nl
            lo, hi, _ = level[t]
            if hi - lo + 1 > minleaf
                @inbounds begin
                    p = work[lo].id
                    u = woff[t]
                    for i in (lo+1):hi
                        wpos[u] = i
                        wpiv[u] = p
                        u += 1
                    end
                end
            end
        end

        # ---- phase B: the level's entire partitioning pass, one parallel map
        @BATCHES getminbatch(ctx, w) scheduler=ctx.scheduler begin
        @BEGINBATCH
            # a batch is single-tasked, so a distance with scratch can hand this batch its
            # own buffers and skip both the locking and the per-call allocation
            bdist = bdists[@batchid()]
        @LOOP for u in 1:w
            i = wpos[u]
            # the caller guarantees these are integer-valued (see BKT's docstring); a
            # fractional distance would round into a bucket it does not belong to
            @inbounds work[i] = IdKey(work[i].id,
                round(Int32, Dist.evaluate(bdist, db[wpiv[u]], db[work[i].id])))
        end
        @END
        end
        cost += w

        # ---- phase C: sort each node's range by key and count what it needs, in parallel
        @BATCHES minbatch scheduler=ctx.scheduler begin
        @BEGINBATCH
            cnt = cntb[@batchid()]   # the batch's grow-only counting-sort histogram
        @LOOP for t in 1:nl
            lo, hi, _ = level[t]
            if hi - lo + 1 <= minleaf
                gcount[t], bcount[t], ncount[t] = 0, hi - lo, 0
            else
                gcount[t], bcount[t], ncount[t] = _countsort!(work, work2, cnt, lo, hi, minleaf)
            end
        end
        @END
        end

        # ---- phase D: serial, but only O(#nodes of the level): turn the counts into
        # disjoint reservations, so phase E never contends for the shared arrays.
        c = Int32(length(bkt.childkey) + 1)
        b = Int32(length(bkt.bucket) + 1)
        x = one(Int32)
        for t in 1:nl
            coff[t], boff[t], noff[t] = c, b, x
            c += gcount[t]
            b += bcount[t]
            x += ncount[t]
        end

        resize!(bkt.childkey, c - 1)
        resize!(bkt.childnode, c - 1)
        resize!(bkt.bucket, b - 1)
        resize!(next, x - 1)

        # ---- phase E: fill the reservations, in parallel again
        @BATCHES minbatch scheduler=ctx.scheduler for t in 1:nl
            lo, hi, slot = level[t]
            _emit!(bkt, work, lo, hi, slot, coff[t], boff[t], noff[t], next, minleaf)
        end

        level, next = next, level
        empty!(next)
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
