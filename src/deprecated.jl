# This file is a part of SimilaritySearch.jl

searchbatch(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer) = searchbatch(idx, getcontext(idx), Q, k)
search(idx::AbstractSearchIndex, q, res::KnnResult) = search(idx, getcontext(idx), q)

push_item!(idx::AbstractSearchIndex, u) = push_item!(idx, getcontext(idx), u)
append_items!(idx::AbstractSearchIndex, u::AbstractDatabase) = append_items!(idx, getcontext(idx), u)

allknn(idx::AbstractSearchIndex, k::Integer) = allknn(idx, getcontext(idx), k)
neardup(idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real; kwargs...) = neardup(idx, getcontext(idx), ϵ; kwargs...)
closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; kwargs) = closestpair(idx, getcontext(idex); kwargs...)

