# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Performance, StatsKnn, StatsComparison, scores, probe

struct StatsKnn
    distancessum::Float64
    nearestdist::Float64
    farthestdist::Float64
    length::Float64

    StatsKnn(res::KnnResult) = new(sum(p.dist for p in res), first(res).dist, last(res).dist, length(res))
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

struct StatsComparison
    macrorecall::Float64
    macroprecision::Float64
    macrof1::Float64
    searchtime::Float64
    stats::StatsKnn
    goldsearchtime::Float64
    goldstats::StatsKnn
end

struct Performance{DataType<:AbstractVector}
    queries::DataType
    ksearch::Int
    popnearest::Bool
    goldreslist::Vector{KnnResult}
    goldsearchtime::Float64
    goldstats::StatsKnn
end

function perf_search_batch(index::AbstractSearchContext, queries, ksearch::Integer, popnearest::Bool)
    m = length(queries)
    if popnearest
        ksearch += 1
    end
    reslist = [KnnResult(ksearch) for i in 1:m]
    start = time()
    for i in 1:m
        search(index, queries[i], reslist[i])
        popnearest && popfirst!(reslist[i])
    end
    elapsed = time() - start
    reslist, elapsed / m
end

function Performance(goldsearch::AbstractSearchContext, queries::AbstractVector, ksearch::Integer; popnearest=false)
    gold, searchtime = perf_search_batch(goldsearch, queries, ksearch, popnearest)
    Performance(queries, ksearch, popnearest, gold, searchtime, StatsKnn(gold))
end

function probe(perf::Performance, index::AbstractSearchContext)
    reslist, searchtime = perf_search_batch(index, perf.queries, perf.ksearch, perf.popnearest)
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

    StatsComparison(recall/n, precision/n, f1/n, searchtime, StatsKnn(reslist), perf.goldsearchtime, perf.goldstats)
end

function scores(gold::Set, res::Set)
    tp = intersect(gold, res)  # tn
    fp = length(setdiff(res, tp)) # |fp| == |ft| when |res| == |gold|
    fn = length(setdiff(gold, tp))
    _tp = length(tp)
    recall = _tp / (_tp + fn)
    precision = _tp / (_tp + fp)

    (recall=recall, precision=precision, f1=2 * precision * recall / (precision + recall))
end

scores(gold::KnnResult, res::KnnResult) = scores(Set(item.id for item in gold), Set(item.id for item in res))
