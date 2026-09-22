# This file is a part of SimilaritySearch.jl

export SketchedSearch

"""
    SketchedSearch{QS,IDX,DIST,DB} <: AbstractSearchIndex

    SketchedSearch(model, nbits::Int, dist::PreMetric, db::AbstractDatabase;
                   factor::Int=8, index=exhaustivesketchindex, minbatch::Int=4, kwargs...)

The whole sketch-based search pipeline behind one ordinary index: it sketches `db` with
`model` at `nbits` bits per component (see [`QuantSketch`](@ref)), indexes the sketches,
and answers a query in the *original* space by sketching it the same way, collecting
`factor * k` candidates under the cheap sketch distance, and re-scoring exactly those with
`dist`.

Assembling that by hand is only a handful of lines, but every one of them is a place to
get it wrong silently: sketching the query with a freshly-fitted encoder instead of the
dataset's (incomparable codes), indexing the sketches under the original distance instead
of [`distance`](@ref)`(qs)`, or forgetting the re-scoring pass and returning the sketch
distances as if they were real ones. None of those raise an error -- they just quietly
return worse results. Going through `SketchedSearch` makes them unrepresentable.

`SketchedSearch` is built once over a fixed `db` and is not incremental: its encoder's
quantization range (and, for the hyperplane models, its anchors) are fitted on exactly the
data given here, so growing the collection means rebuilding it rather than pushing into it.

Because it is an `AbstractSearchIndex`, [`search`](@ref)/[`searchbatch`](@ref) work on it
unchanged, and the ids and distances it returns are ids into `db` and true `dist` values,
so it is a drop-in replacement for an exact index -- swapping one in is the whole
experiment. `factor` is the knob that trades recall for time: the sketch stage is fast but
lossy, so it is asked for more candidates than needed and the exact distance settles the
final order among them.

# Arguments
- `model`: the sketch model; anything [`sketchvalues!`](@ref) accepts (a hyperplane model
  like [`DistantHyperplanes`](@ref), or a rotation like [`RandomProjections`](@ref))
- `nbits`: bits per component -- `1` (plain [`bitsketch`](@ref) + Hamming), `2`, `4` or `8`
- `dist`: the original, exact distance, used to re-score candidates
- `db`: the original database

# Keyword Arguments
- `factor`: how many times `k` candidates the sketch stage retrieves before re-scoring;
  `1` disables the widening (but still re-scores, so distances stay exact)
- `index`: `(sketchdist, sketches) -> AbstractSearchIndex`, building the index over the
  sketches. Defaults to an [`ExhaustiveSearch`](@ref) -- a linear scan, which is the point:
  it is over sketches many times smaller than the objects. Pass a closure returning a
  *ready-to-search* index (already `index!`-ed) to use e.g. a `SearchGraph` instead.
- `minbatch`: minimum number of objects per parallel task while sketching `db`
- remaining keywords are forwarded to [`QuantSketch`](@ref) (`normalize`, `minmax`,
  `quant`, `samplesize`)

# Examples

```julia
julia> using SimilaritySearch, SimilaritySearch.Projections

julia> dist = SimilaritySearch.Dist.L2();

julia> db = MatrixDatabase(rand(Float32, 8, 10_000));

julia> Q  = MatrixDatabase(rand(Float32, 8, 100));

julia> m = DistantHyperplanes(dist, db, 128; verbose=false);

julia> S = SketchedSearch(m, 4, dist, db);          # 4-bit codes, 8x candidate widening

julia> ids, dists = searchbatch(S, GenericContext(), Q, 10);   # ids into db, exact dists
```
"""
struct SketchedSearch{QS<:QuantSketch,IDX<:AbstractSearchIndex,DIST<:PreMetric,DB<:AbstractDatabase} <: AbstractSearchIndex
    qs::QS
    sketches::IDX
    dist::DIST
    db::DB
    factor::Int
end

"""
    exhaustivesketchindex(sketchdist, sketches) -> ExhaustiveSearch

The default `index` builder of [`SketchedSearch`](@ref): a plain [`ExhaustiveSearch`](@ref)
over the sketches. A linear scan is a sensible default here precisely because the sketches
are far smaller than the objects they stand for, and it keeps the recall of the pipeline
attributable to the sketch alone -- no second layer of approximation on top.
"""
exhaustivesketchindex(sketchdist, sketches) = ExhaustiveSearch(sketchdist, sketches)

function SketchedSearch(model, nbits::Int, dist::PreMetric, db::AbstractDatabase;
        factor::Int=8,
        index=exhaustivesketchindex,
        minbatch::Int=4,
        kwargs...)
    factor >= 1 || throw(ArgumentError("SketchedSearch: factor=$factor must be >= 1"))
    qs = QuantSketch(model, nbits, db; kwargs...)
    B = quantsketch(qs, db; minbatch)
    SketchedSearch(qs, index(distance(qs), B), dist, db, factor)
end

"""
    distance(S::SketchedSearch)

The *original* distance `S` re-scores its candidates with -- the one its results are
expressed in. The cheap sketch-space distance is `distance(S.qs)` instead; see
[`QuantSketch`](@ref).
"""
@inline distance(S::SketchedSearch) = S.dist

@inline database(S::SketchedSearch) = S.db
@inline database(S::SketchedSearch, i::Integer) = S.db[i]
@inline Base.length(S::SketchedSearch) = length(S.db)

"""
    search(S::SketchedSearch, ctx::AbstractContext, q, res::AbstractKnnQueue) -> res

Solves `q` in two stages: sketches it with `S`'s encoder and retrieves
`factor * maxlength(res)` candidates from the sketch index under the sketch distance, then
re-scores every candidate with the original distance against the original objects and
pushes the results into `res`. Only the second stage's distances are charged to `ctx`'s
evaluation counter by this method; the sketch stage charges its own.

# Arguments
- `S`: the pipeline
- `ctx`: the running context
- `q`: the query, in the *original* space (not pre-sketched)
- `res`: the result set, which receives ids into `database(S)` and exact `distance(S)` values
"""
function search(S::SketchedSearch, ctx::AbstractContext, q, res::AbstractKnnQueue)
    cand = knnqueue(KnnSorted, S.factor * maxlength(res))
    search(S.sketches, ctx, quantsketch(S.qs, q), cand)
    dist, db = S.dist, S.db
    n = 0
    for p in cand
        push_item!(res, p.id, evaluate(dist, db[p.id], q))
        n += 1
    end

    add_distance_evaluations!(ctx, n)
    res
end
