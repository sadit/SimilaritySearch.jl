# This file is part of SimilaritySearch.jl


export InformativeLog, LOG

"""
    InformativeLog(; append_prob=0.01, push_prob=0.0001)

Informative logger. It generates an output with some given probability
"""
Base.@kwdef struct InformativeLog
    append_prob = 0.01
    push_prob = 0.0001
end

function LOG(logger::InformativeLog, ::typeof(append_items!), index::AbstractSearchIndex, sp::Integer, ep::Integer, n::Integer)
    rand() < logger.append_prob && println(stderr, "append_items ", (sp=sp, ep=ep, n=n), " ", Dates.now())
end

function LOG(logger::InformativeLog, ::typeof(push_item!), index::AbstractSearchIndex, n::Integer)
    if rand() < logger.push_prob
        println(stderr, "push_item n=$(length(index)), $(string(index.algo)), $(Dates.now())")
    end
end
