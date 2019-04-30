#  Copyright 2016-2019  Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

export Performance, PerformanceResult, probe
using Statistics: mean

mutable struct PerformanceResult
    precision::Float64
    recall::Float64
    macrof1::Float64
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

    function Performance(dist::Function, db::AbstractVector{T}, queries::AbstractVector{T}; queries_from_db=false, expected_k=10, create_index=Sequential) where T
        results = Vector{Set{Int}}(undef, length(queries))
        distances_sum = Vector{Float64}(undef, length(queries))

        s = create_index(db)
        start = time()
        for i in 1:length(queries)
            res = search(s, dist, queries[i], KnnResult(expected_k))
            if queries_from_db
                popfirst!(res)
            end
            results[i] = Set(item.objID for item in res)
            distances_sum[i] = sum(item.dist for item in res)
        end

        elapsed = time() - start
        new{T}(db, queries, results, distances_sum, expected_k, queries_from_db, elapsed / length(queries))
    end

    function Performance(db::AbstractVector{T}, dist::Function; num_queries::Int=128, expected_k::Int=10) where T
        queries = rand(db, num_queries)
        Performance(dist, db, queries, queries_from_db=true, expected_k=expected_k)
    end
end

function probe(perf::Performance, index::Index, dist::Function; repeat::Int=1, aggregation=:mean, field=:seconds)
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

function _probe(perf::Performance, index::Index, dist::Function)
    eval_counter = 0
    function dist_(a_, b_)
        eval_counter += 1
        dist(a_, b_)
    end

    p = PerformanceResult()
    m = length(perf.queries)
    p.evaluations = eval_counter
    p.seconds = 0.0
    p.distances_sum = 0.0

    for i in 1:m
        start = time()
        res = search(index, dist_, perf.queries[i], KnnResult(perf.expected_k))
        p.seconds += time() - start
        if perf.queries_from_db
            popfirst!(res)
        end
        base = perf.results[i]
        curr = Set(item.objID for item in res)
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
        p.macrof1 += 2 * precision * recall / (precision + recall)
    end

    p.evaluations = (eval_counter - p.evaluations) / m
    p.seconds = p.seconds / m
    p.precision /= m
    p.recall /= m
    p.macrof1 /= m
    p.exhaustive_search_seconds = perf.exhaustive_search_seconds

    p
end
