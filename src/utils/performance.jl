# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Performance, PerformanceResult, probe
using StatsBase: mean

mutable struct PerformanceResult
    precision::Float64
    recall::Float64
    f1::Float64
    seconds::Float64
    exhaustive_search_seconds::Float64
    evaluations::Float64
    distances_sum::Float64
    PerformanceResult() = new(0, 0, 0, 0, 0, 0)
end

mutable struct Performance{T}
    db::AbstractVector{T}
    queries::AbstractVector{T}
    results::Vector{Set{Int}}
    distances_sum::Vector{Float64}
    expected_k::Int
    queries_from_db::Bool
    exhaustive_search_seconds::Float64

    function Performance(dist::PreMetric, db::AbstractVector{T}, queries::AbstractVector{T}; queries_from_db=false, expected_k=10, create_index=Sequential) where T
        results = Vector{Set{Int}}(undef, length(queries))
        distances_sum = Vector{Float64}(undef, length(queries))

        s = create_index(db)
        start = time()
        res = KnnResult(expected_k)
        for i in eachindex(queries)
            empty!(res)
            search(s, dist, queries[i], res)
            if queries_from_db
                popfirst!(res)
            end
            results[i] = Set(item.id for item in res)
            distances_sum[i] = sum(item.dist for item in res)
        end

        elapsed = time() - start
        new{T}(db, queries, results, distances_sum, expected_k, queries_from_db, elapsed / length(queries))
    end

    function Performance(db::AbstractVector{T}, dist::PreMetric; num_queries::Int=128, expected_k::Int=10) where T
        queries = rand(db, num_queries)
        Performance(dist, db, queries, queries_from_db=true, expected_k=expected_k)
    end
end

function probe(perf::Performance, index::Index, dist::PreMetric; repeat::Int=1, aggregation=:mean, field=:seconds)
    if repeat == 1
        return _probe(perf, index, dist)
    end

    if aggregation == :mean
        p = _probe(perf, index, dist)
        for i in 2:repeat
            q = _probe(perf, index, dist)
            p.recall += q.recall
            p.seconds += q.seconds
            p.evaluations += q.evaluations
        end
        p.recall /= repeat
        p.seconds /= repeat
        p.evaluations /= repeat
        return p
    end

    M = [_probe(perf, index, dist) for i in 1:repeat]
    sort!(M, by=(x) -> getfield(x, field))

    if aggregation == :median
        return M[ceil(Int, length(M) / 2)]
    elseif aggregation == :min
        return M[1]
    elseif aggregation == :max
        return M[end]
    else
        error("Unknown aggregation strategy: $aggregation")
    end
end

mutable struct DistanceCounter{T<:PreMetric} <: PreMetric
    eval_counter::Int
    dist::T
end

function DistanceCounter(dist::T) where {T<:PreMetric}
    DistanceCounter(0, dist)
end

function evaluate(d::DistanceCounter, a, b)
    d.eval_counter += 1
    evaluate(d.dist, a, b)
end


function _probe(perf::Performance, index::Index, dist::PreMetric)
    dist_ = DistanceCounter(dist)

    p = PerformanceResult()
    m = length(perf.queries)
    p.evaluations = dist_.eval_counter
    p.seconds = 0.0
    p.distances_sum = 0.0

    res = KnnResult(perf.expected_k)
    for i in 1:m
        empty!(res)
        start = time()
        res = search(index, dist_, perf.queries[i], res)
        p.seconds += time() - start
        if perf.queries_from_db
            popfirst!(res)
        end
        base = perf.results[i]
        curr = Set(item.id for item in res)
        for item in res
            p.distances_sum += item.dist
        end
        tp = intersect(base, curr)  # tn
        fp = length(setdiff(curr, tp)) # |fp| == |ft| when |curr| == |base|
        fn = length(setdiff(base, tp))
        _tp = length(tp)
        precision = _tp / (_tp + fp)
        recall = _tp / (_tp + fn)
        p.precision += precision
        p.recall += recall
        p.f1 += 2 * precision * recall / (precision + recall)
    end

    p.evaluations = (dist_.eval_counter - p.evaluations) / m
    p.seconds = p.seconds / m
    p.precision /= m
    p.recall /= m
    p.f1 /= m
    p.exhaustive_search_seconds = perf.exhaustive_search_seconds

    p
end
