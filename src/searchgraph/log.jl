# This file is a part of SimilaritySearch.jl

function LOG(logger::InformativeLog, ::typeof(push_item!), index::SearchGraph, n::Integer)
    if rand() < logger.push_prob
        N = neighbors(index.adj, n)
        println(stderr, "push_item! n=$(length(index)), neighborhood=$(length(N)), $(string(index.algo)), $(Dates.now())")
    end
end

function LOG(logger::InformativeLog, ::typeof(append_items!), index::SearchGraph, sp::Integer, ep::Integer, n::Integer)
    if rand() < logger.append_prob
        println(stderr, "append_items! sp=$sp, ep=$ep, n=$(length(index)), $(string(index.algo)), $(Dates.now())")
    end
end
