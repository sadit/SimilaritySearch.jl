# This file is a part of SimilaritySearch.jl

export Performance, probe, scores, recall_score

statsknn(res::KnnResult) = (
    maximum=maximum(res),
    minimum=minimum(res),
    k=length(res)
)

statsknn(reslist::AbstractVector{<:KnnResult}) = (
    maximum=mean(maximum(res) for res in reslist),
    minimum=mean(minimum(res) for res in reslist),
    k=mean(length(res) for res in reslist)
)

"""
    Performance(_goldsearch::AbstractSearchContext, queries::AbstractVector, ksearch::Integer; popnearest=false)

Creates performance comparer for the given set of queries using a gold standard index.

- `_goldsearch`: a gold standard index
- `queries`: a set of queries
- `ksearch`: the number of neighbors to retrieve
- `popnearest`: set as `true` whenever queries are part of the dataset.
"""
struct Performance
    queries
    ksearch::Int
    popnearest::Bool
    gold
    goldsearchtime::Float64
    goldstats
end

function Performance(goldindex::AbstractSearchContext, queries, ksearch::Integer; popnearest=false)
    gold, searchtime = perf_searchbatch(goldindex, queries, ksearch, popnearest)
    Performance(queries, Int(ksearch), popnearest, [Set(keys(res)) for res in gold], searchtime, statsknn(gold))
end

function perf_searchbatch(index::AbstractSearchContext, queries, ksearch::Integer, popnearest::Bool)
    m = length(queries)
    if popnearest
        ksearch += 1
    end
    reslist = [KnnResult(ksearch) for i in 1:m]
    p = @timed searchbatch(index, queries, reslist; parallel=false)
    if popnearest
        for r in reslist
            popfirst!(r)
        end
    end

    @show m, p.time / m, p.bytes / m, p.gctime, p.gcstats
    reslist, p.time / m
end

"""
    probe(perf::Performance, _index::AbstractSearchContext)

Compares the performance of `_index` with the gold standard index of `perf`.
"""
function probe(perf::Performance, index::AbstractSearchContext)
    reslist, searchtime = perf_searchbatch(index, perf.queries, perf.ksearch, perf.popnearest)
    
    recall = mean(recall_score.(perf.gold, reslist))
    (
        recall=recall,
        searchtime=searchtime,
        qps=1/searchtime,
        statsknn(reslist)...,
        goldsearchtime=perf.goldsearchtime,
        goldqps=1/perf.goldsearchtime,
        speeup=perf.goldsearchtime/searchtime,
        perf.goldstats...
    )
end

"""
    scores(gold, res)

Compute recall and precision scores from the result sets.
"""
function scores(gold::Set, res)
    tp = intersect(gold, res)  # tn
    fp = length(setdiff(res, tp)) # |fp| == |ft| when |res| == |gold|
    fn = length(setdiff(gold, tp))
    _tp = length(tp)
    recall = _tp / (_tp + fn)
    precision = _tp / (_tp + fp)

    (recall=recall, precision=precision, f1=2 * precision * recall / (precision + recall))
end

scores(gold::Set, res::KnnResult) = scores(gold, keys(res))
scores(gold::KnnResult, res) = scores(Set(keys(gold)), res)
recall_score(gold, res) = scores(gold, res).recall
