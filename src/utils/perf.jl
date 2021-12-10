# This file is a part of SimilaritySearch.jl

export Performance, probe, recallscore, timedsearchbatch, macrorecall

statsknn(reslist::AbstractVector{<:KnnResult}) = (
    maximum=mean(maximum(res) for res in reslist),
    minimum=mean(minimum(res) for res in reslist),
    k=mean(length(res) for res in reslist)
)

"""
    recallscore(gold::Set, res)

Compute recall and precision scores from the result sets.
"""
function recallscore(gold, res)::Float64
    length(intersect(_convert_as_set(gold), _convert_as_set(res))) / length(gold)
end

_convert_as_set(a::KnnResult) = keys(a)
_convert_as_set(a::Set) = a
_convert_as_set(a::Vector{<:Integer}) = a

function macrorecall(gold, res)::Float64
    s = 0.0
    for i in eachindex(gold)
        s += recallscore(gold[i], res[i])
    end

    s / length(gold)
end

function timedsearchbatch(index, queries, ksearch::Integer; parallel=false)
    reslist = knnresults(ksearch, length(queries))
    t = @elapsed searchbatch(index, queries, reslist; parallel)
    reslist, t/length(reslist)
end