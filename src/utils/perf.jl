# This file is a part of SimilaritySearch.jl

export Performance, StatsKnn, StatsComparison, scores, probe

mutable struct DistCounter{DistType<:PreMetric} <: PreMetric
    dist::DistType
    count::Int
end

function evaluate(D::DistCounter, a, b)
    D.count += 1
    evaluate(D.dist, a, b)
end


"""
    StatsKnn(res::KnnResult)
    StatsKnn(res::AbstractVector)

Computes some basic statistics for the KnnResult or an array of KnnResults
"""
struct StatsKnn
    distancessum::Float64
    nearestdist::Float64
    farthestdist::Float64
    length::Float64

    StatsKnn(res::KnnResult) = new(sum(p[end] for p in res), minimum(res), maximum(res), length(res))
    function StatsKnn(reslist::AbstractVector{T}) where {T<:KnnResult}
        distancessum = 0.0
        nearestdist = 0.0
        farthestdist = 0.0
        len = 0.0

        for res in reslist
            p = StatsKnn(res)
            distancessum += p.distancessum
            nearestdist += p.nearestdist
            farthestdist += p.farthestdist
            len += p.length
        end

        n = length(reslist)
        new(distancessum/n, nearestdist/n, farthestdist/n, len/n)
    end
end

"""
    StatsComparison

Stores the result of the comparison between two indexes, see [`Performance`](@ref).
"""
struct StatsComparison
    macrorecall::Float64
    macroprecision::Float64
    macrof1::Float64
    searchtime::Float64
    evaluations::Float64
    stats::StatsKnn
    goldsearchtime::Float64
    goldstats::StatsKnn
end

"""
    Performance(_goldsearch::AbstractSearchContext, queries::AbstractVector, ksearch::Integer; popnearest=false)

Creates performance comparer for the given set of queries using a gold standard index.

- `_goldsearch`: a gold standard index
- `queries`: a set of queries
- `ksearch`: the number of neighbors to retrieve
- `popnearest`: set as `true` whenever queries are part of the dataset.
"""
struct Performance{DataType<:AbstractVector, KnnResultType<:KnnResult}
    queries::DataType
    ksearch::Int
    popnearest::Bool
    goldreslist::Vector{KnnResultType}
    goldsearchtime::Float64
    goldevaluations::Float64
    goldstats::StatsKnn
end

function Performance(_goldsearch::AbstractSearchContext, queries::AbstractVector, ksearch::Integer; popnearest=false)
    dist = DistCounter(_goldsearch.dist, 0)
    goldsearch = copy(_goldsearch, dist=dist)
    gold, searchtime, evaluations = perf_search_batch(goldsearch, queries, ksearch, popnearest)
    Performance(queries, Int(ksearch), popnearest, gold, searchtime, evaluations, StatsKnn(gold))
end


function perf_search_batch(index::AbstractSearchContext, queries, ksearch::Integer, popnearest::Bool)
    m = length(queries)
    if popnearest
        ksearch += 1
    end
    reslist = [KnnResult(ksearch) for i in 1:m]
    search(index, queries[1], ksearch) # warming step
    evaluations = index.dist.count
    start = time()

    for i in 1:m
        search(index, queries[i], reslist[i])
        popnearest && popfirst!(reslist[i])
    end

    elapsed = time() - start
    reslist, elapsed / m, (index.dist.count - evaluations) / m
end

"""
    probe(perf::Performance, _index::AbstractSearchContext)

Compares the performance of `_index` with the gold standard index of `perf`.
"""
function probe(perf::Performance, _index::AbstractSearchContext)
    index = copy(_index, dist=DistCounter(_index.dist, 0))
    reslist, searchtime, evaluations = perf_search_batch(index, perf.queries, perf.ksearch, perf.popnearest)
    n = length(reslist)
    recall = 0.0
    nearest = 0.0
    precision = 0.0
    f1 = 0.0
    for i in eachindex(reslist)
        p = scores(perf.goldreslist[i], reslist[i])
        recall += p.recall
        precision += p.precision
        f1 += p.f1
    end

    StatsComparison(recall/n, precision/n, f1/n, searchtime, evaluations, StatsKnn(reslist), perf.goldsearchtime, perf.goldstats)
end

"""
    scores(gold, res)

Compute recall and precision scores from the result sets.
"""
function scores(gold::Set, res::Set)
    tp = intersect(gold, res)  # tn
    fp = length(setdiff(res, tp)) # |fp| == |ft| when |res| == |gold|
    fn = length(setdiff(gold, tp))
    _tp = length(tp)
    recall = _tp / (_tp + fn)
    precision = _tp / (_tp + fp)

    (recall=recall, precision=precision, f1=2 * precision * recall / (precision + recall))
end

scores(gold::KnnResult, res::KnnResult) = scores(Set(gold.id), Set(res.id))
