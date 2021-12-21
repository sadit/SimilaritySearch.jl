# This file is a part of SimilaritySearch.jl

export Performance, probe, recallscore, timedsearchbatch, macrorecall

"""
    recallscore(gold::Set, res)

Compute recall and precision scores from the result sets.
"""
function recallscore(gold, res)::Float64
    length(intersect(_convert_as_set(gold), _convert_as_set(res))) / length(gold)
end

_convert_as_set(a::Set) = a
_convert_as_set(a::AbstractVector) = Set(a)
_convert_as_set(a::KnnResult) = Set(a.id)

function macrorecall(goldI::AbstractMatrix, resI::AbstractMatrix, k=size(goldI, 1))::Float64
    @assert size(goldI) == size(resI)
    n = size(goldI, 2)
    s = 0.0
    for i in 1:n
        g = view(goldI, 1:k, i)
        r = view(resI, 1:k, i)
        s += recallscore(g, r)
    end

    s / n
end

function macrorecall(goldlist::AbstractVector, reslist::AbstractVector)::Float64
    @assert size(goldlist) == size(reslist)
    s = 0.0
    n = length(goldlist)
    for i in 1:n
        g = goldlist[i]
        r = reslist[i]
        #g = view(goldlist[i], 1:k)
        #r = view(reslist[i], 1:k)
        s += recallscore(g, r)
    end

    s / n
end

function timedsearchbatch(index, Q, ksearch::Integer; parallel=false)
    m = length(Q)
    I = zeros(Int32, ksearch, m)
    D = Matrix{Float32}(undef, ksearch, m)
    t = @elapsed (I, D = searchbatch(index, Q, I, D; parallel))
    I, D, t/length(Q)
end