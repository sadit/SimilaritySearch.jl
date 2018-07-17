#  Copyright 2016  Eric S. Tellez <eric.tellez@infotec.mx>
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
    recall::Float64
    seconds::Float64
    distances::Float64
end

mutable struct Performance{T}
    db::AbstractVector{T}
    querySet::AbstractVector{T}
    results::Vector{Result}
    expected_k::Int
    shift_expected_k::Int
    seqtime::Float64
end

function Performance(db::AbstractVector{T}, dist::D, querySet::AbstractVector{T}; expected_k::Int=10) where {T,D}
    results = Vector{Result}(undef, length(querySet))
    s = Sequential(db, dist)

    start_time = time()
    for i in 1:length(querySet)
        results[i] = search(s, querySet[i], KnnResult(expected_k))
    end

    return Performance(db, querySet, results, expected_k, 0, (time() - start_time) / length(querySet))
end

function Performance(db::AbstractVector{T}, dist::D; numqueries::Int=128, expected_k::Int=10) where {T,D}
    querySet = rand(db, numqueries)
    Performance(db, dist, querySet, expected_k=expected_k)
end

function probe(perf::Performance, index::Index; repeat::Int=1, aggregation=:mean, field=:seconds, use_distances::Bool=false)
    if repeat == 1
        return _probe(perf, index, use_distances=use_distances)
    end

    if aggregation == :mean
        p = _probe(perf, index, use_distances=use_distances)
        for i in 2:repeat
            q = _probe(perf, index, use_distances=use_distances)
            p.recall += q.recall
            p.seconds += q.seconds
            p.distances += q.distances
        end
        p.recall /= repeat
        p.seconds /= repeat
        p.distances /= repeat
        return p
    end

    M = [_probe(perf, index, use_distances=use_distances) for i in 1:repeat]
    sort!(M, by=(x) -> getfield(x, field))

    if aggregation == :median
        return M[ceil(Int, length(M) / 2)]
    elseif aggregation == :min
        return M[1]
    elseif aggregation == :max
        return M[end]
    else
        error("Unknown aggregation strategy $aggregation")
    end
end

function _probe(perf::Performance, index::Index; use_distances::Bool=false)
    p::PerformanceResult = PerformanceResult(0.0, 0.0, 0.0)
    m::Int = length(perf.querySet)
    tlist = Vector{Float64}(undef, m)
    rlist = Vector{Float64}(undef, m)
    dlist = Vector{Int}(undef, m)

    counting_calls = :calls in fieldnames(typeof(index.dist))

    for i = 1:m
        q = perf.querySet[i]
        if counting_calls
          start_calls = index.dist.calls
        else
          start_calls = 0
        end

        start_time = time()
        res = search(index, q, KnnResult(perf.expected_k))
        tlist[i] = time() - start_time
        if counting_calls
          dlist[i] = index.dist.calls - start_calls
        else
          dlist[i] = 0
        end

        # p.seconds += time() - start_time
        # p.distances += index.dist.calls - start_calls

        base_res = perf.results[i]
        if use_distances
            rlist[i] = recallByDist(base_res, res, shift=perf.shift_expected_k)
        else
            rlist[i] = recall(base_res, res, shift=perf.shift_expected_k)
        end
    end

    p.recall = mean(rlist)
    # p.recall = mean(rlist) - std(rlist)
    p.seconds = mean(tlist)
    p.distances = mean(dlist)
    return p
end
