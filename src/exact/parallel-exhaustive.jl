# This file is a part of SimilaritySearch.jl

export ParallelExhaustiveSearch

"""
    struct ParallelExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex

    ParallelExhaustiveSearch(dist::PreMetric, db::AbstractDatabase)
    ParallelExhaustiveSearch(dist::PreMetric, db::AbstractVecOrMat)


A brute-force exact index, like [`ExhaustiveSearch`](@ref), but that solves each query by evaluating `dist`
against every element of `db` in parallel (across `Threads.nthreads()` tasks). Each batch of the underlying
[`@BATCHES`](@ref) call accumulates its own private, lock-free top-k buffer (indexed by `@batchid()`), merged
into the final result once all batches join -- see [`search`](@ref search(::ParallelExhaustiveSearch, ::GenericContext, ::Any, ::AbstractKnn))
for details. Useful as a gold-standard baseline for small-to-medium datasets where parallelizing a single
query is beneficial.

Note that this should not be used in conjunction with `searchbatch(...; parallel=true)` since they will
compete for the same thread pool.

# Arguments
- `dist`: the distance function
- `db`: the database being indexed, given either as an `AbstractDatabase` or as a raw vector/matrix
"""
struct ParallelExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
end


function getcontext(::ParallelExhaustiveSearch)
    GenericContext()
end

"""

    search(pex::ParallelExhaustiveSearch, ctx::GenericContext, q, res::AbstractKnn) -> res

Solves queries evaluating `dist` in parallel for the query and all elements in the dataset.

Solves query `q` by evaluating the distance between `q` and every item of the indexed database in
parallel. Instead of pushing every candidate into the shared `res` under a lock, each batch
accumulates its own private top-`k` buffer (`k = maxlength(res)`), indexed by `@batchid()` -- race-free
by construction, no lock needed -- and all batches' buffers are merged into `res` once they have all
joined (`@END`, run sequentially, once).

The extra memory this needs is `k * @nbatches()` `IdDist` entries: `@nbatches()` never scales with `n`
(the database size) -- [`getminbatch`](@ref) aims for ~8 batches per thread regardless of `n`, and
`@BATCHES`'s own fast path collapses to a single batch entirely whenever `n` is small relative to the
computed `minbatch` -- so this temporary buffer stays bounded by the thread count and `k`, not by the
size of the database being searched. `ctx.maxbatches` (default `8 * nthreads()`, see
[`GenericContext`](@ref)) directly caps `@nbatches()` further, for cases with a large `k` and/or
`nthreads()` where even that bounded buffer is too large; see [`getminbatch`](@ref) for the
trade-offs of capping it (fewer batches can leave threads idle and worsens load-balancing).

# Arguments
- `pex`: the search structure
- `ctx`: the running context; `ctx.maxbatches` bounds the number of batches (and thus the size of
  the temporary `k * @nbatches()` buffer), passed as `getminbatch(ctx, n)`
- `q`: the query to solve
- `res`: the result set that receives the candidates
"""
function search(pex::ParallelExhaustiveSearch, ctx::GenericContext, q, res::AbstractKnn)
    dist = distance(pex)
    n = length(pex)
    k = maxlength(res)
    minbatch = getminbatch(ctx, n)

    # NOTE: forced to :default (never :static/:greedy, regardless of ctx.scheduler) because
    # this per-query search is itself commonly invoked from *within* an outer @BATCHES-
    # parallelized per-query loop (e.g. searchbatch!/allknn/closestpair when `pex` is the
    # given index) -- native `:static` errors ("cannot be used concurrently or nested") in
    # that situation. This loop body has no Threads.threadid()-indexed state at all (each
    # batch only ever touches its own @batchid()-indexed column), so :default's migratable
    # tasks are safe here regardless of the global scheduler. `ctx.scheduler === :sequential`
    # is still honored (falls through to @BATCHES's own single-batch fast path), since that
    # is an explicit request to disable threading entirely, not a scheduler *kind* choice.
    @BATCHES minbatch scheduler=(ctx.scheduler === :sequential ? :sequential : :default) begin
    @BEGIN
        # one private, lock-free top-k buffer per batch; @nbatches() is bounded (~8 * nthreads(),
        # via getminbatch), never by n, so this never grows with the database size
        R = zeros(IdDist, k, @nbatches())
        used = zeros(Int32, @nbatches())
    @BEGINBATCH
        r = knnqueue(KnnSorted, view(R, :, @batchid()))
    @LOOP for i in 1:n
        d = Dist.evaluate(dist, database(pex, i), q)
        push_item!(r, i, d)
    end
    @ENDBATCH
        used[@batchid()] = length(r)
    @END
        for b in 1:@nbatches(), j in 1:used[b]
            push_item!(res, R[j, b])
        end
    end

    add_distance_evaluations!(ctx, length(pex))
    res
end

function push_item!(pex::ParallelExhaustiveSearch, ctx::GenericContext, u)
    push_item!(pex.db, u)
    LOG(ctx.logger, :push_item!, pex, ctx, length(pex), length(pex))
    pex
end

function append_items!(pex::ParallelExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(pex)
    append_items!(pex.db, u)
    ep = length(pex)
    LOG(ctx.logger, :append_items!, pex, ctx, sp, ep)
    pex
end

function index!(pex::ParallelExhaustiveSearch, ctx::GenericContext)
    # do nothing
    LOG(ctx.logger, :index!, pex, ctx, length(pex), length(pex))
    pex
end
