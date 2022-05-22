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
_convert_as_set(a::AbstractKnnResult) = Set(idview(a))

"""
    macrorecall(goldI::AbstractMatrix, resI::AbstractMatrix, k=size(goldI, 1))::Float64

Computes the macro recall score using goldI as gold standard and resI as predictions;
it expects that matrices of integers (identifiers). If `k` is given, then the results are cut to first `k`.
"""
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

"""
    macrorecall(goldlist::AbstractVector, reslist::AbstractVector)::Float64

Computes the macro recall score using sets of results (AbstractKnnResult objects or vectors of itegers).
"""
function macrorecall(goldlist::AbstractVector, reslist::AbstractVector)::Float64
    @assert size(goldlist) == size(reslist)
    s = 0.0
    n = length(goldlist)
    for i in 1:n
        g = goldlist[i]
        r = reslist[i]
        s += recallscore(g, r)
    end

    s / n
end
