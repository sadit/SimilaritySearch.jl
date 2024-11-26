# This file is a part of SimilaritySearch.jl

#=
searchbatch(idx::AbstractSearchIndex, Q::AbstractDatabase, k::Integer) = searchbatch(idx, getcontext(idx), Q, k)
search(idx::AbstractSearchIndex, q, res::KnnResult) = search(idx, getcontext(idx), q, res)

push_item!(idx::AbstractSearchIndex, u) = push_item!(idx, getcontext(idx), u)
append_items!(idx::AbstractSearchIndex, u::AbstractDatabase) = append_items!(idx, getcontext(idx), u)
index!(idx::AbstractSearchIndex) = index!(idx, getcontext(idx))

allknn(idx::AbstractSearchIndex, k::Integer) = allknn(idx, getcontext(idx), k)
neardup(idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real; kwargs...) = neardup(idx, getcontext(idx), ϵ; kwargs...)
closestpair(idx::AbstractSearchIndex; kwargs) = closestpair(idx, getcontext(idx); kwargs...)

optimize_index!(idx::AbstractSearchIndex, kind::ErrorFunction=MinRecall(0.9); kwargs...) = optimize_index!(idx, getcontext(idx), kind; kwargs...)

=#
