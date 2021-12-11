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

function macrorecall(goldI::Matrix, resI::Matrix, k=size(goldI, 1))::Float64
    n = size(goldI, 2)
    s = 0.0
    for i in 1:n
        g = view(goldI, 1:k, i)
        r = view(resI, 1:k, i)
        s += recallscore(g, r)
    end

    s / n
end

function timedsearchbatch(index, Q, ksearch::Integer; parallel=false)
    m = length(Q)
    I = zeros(Int32, ksearch, m)
    D = Matrix{Float32}(undef, ksearch, m)

    t = @elapsed if parallel
        Threads.@threads for i in eachindex(Q)
            @inbounds search(index, Q[i], KnnResult(I, D, i))
        end
    else
        @inbounds for i in eachindex(Q)
            @elapsed search(index, Q[i], KnnResult(I, D, i))
        end
    end

    I, D, t/length(Q)
end