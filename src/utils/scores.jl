# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export scores, knnstats

function scores(gold::Set, res::Set)
    tp = intersect(gold, res)  # tn
    fp = length(setdiff(res, tp)) # |fp| == |ft| when |res| == |gold|
    fn = length(setdiff(gold, tp))
    _tp = length(tp)
    recall = _tp / (_tp + fn)
    precision = _tp / (_tp + fp)

    (recall=recall, precision=precision, f1=2 * precision * recall / (precision + recall))
end

function knnstats(res::KnnResult)
    (
        distances_sum = sum(p.dist for p in res),
        nearest_dist = first(res).dist,
        farthest_dist = last(res).dist,
        length = length(res)
    )
end

function knnstats(reslist::AbstractVector)
    distances_sum = 0.0
    nearest_dist = 0.0
    farthest_dist = 0.0
    len = 0.0

    for res in reslist
        p = knnstats(res)
        distances_sum += p.distances_sum
        nearest_dist += p.nearest_dist
        farthest_dist += p.farthest_dist
        len += p.length
    end

    n = length(reslist)

    (
        avg_distances_sum = distances_sum/n,
        avg_nearest_dist = nearest_dist/n,
        avg_farthest_dist = farthest_dist/n,
        avg_length = len/n,
    )
end

function scores(gold::KnnResult, res::KnnResult)
    scores(Set(item.id for item in gold), Set(item.id for item in res))
end

function scores(scorelist::AbstractVector{S}) where {S<:NamedTuple}
    n = length(scorelist)
    recall = 0.0
    nearest = 0.0
    precision = 0.0
    f1 = 0.0
    for p in scorelist
        recall += p.recall
        precision += p.precision
        f1 += p.f1
    end

    (macro_recall=recall/n, macro_precision=precision/n, macro_f1=f1/n)
end
