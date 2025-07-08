# This file is a part of SimilaritySearch.jl

export recallscore, macrorecall

"""
    recallscore(gold::Set, res)

Compute recall and precision scores from the result sets.
"""
function recallscore(gold, res)::Float64
    length(intersect(idset(gold), idset(res))) / length(gold)
end

idset(a::Set) = a
idset(a::AbstractVector) = Set(a)
idset(res::Knn) = Set(IdView(res))
idset(res::XKnn) = Set(IdView(res))

"""
    macrorecall(goldI::AbstractMatrix, resI::AbstractMatrix, k=size(goldI, 1))::Float64

Computes the macro recall score using goldI as gold standard and resI as predictions;
it expects that matrices of integers (identifiers). If `k` is given, then the results are cut to first `k`.
"""

function macrorecall(goldI::AbstractMatrix, resI::AbstractMatrix, k::Integer=size(goldI, 1))::Float64
    n = size(goldI, 2)
    s = 0.0
    for i in 1:n
        s += recallscore(view(goldI, 1:k, i), view(resI, 1:k, i))
    end

    s / n
end

"""
    macrorecall(goldlist::AbstractVector, reslist::AbstractVector)::Float64

Computes the macro recall score using sets of results (Knn objects or vectors of itegers).
"""
function macrorecall(goldlist::AbstractVector, reslist::AbstractVector)::Float64
    @assert length(goldlist) == length(reslist) "$(length(goldlist)) == $(length(reslist))"
    s = 0.0
    n = length(goldlist)
    for i in 1:n
        g = goldlist[i]
        r = reslist[i]
        s += recallscore(g, r)
    end

    s / n
end
