# This file is part of InvertedFiles.jl

struct BinInvFileOutput{InvFileType<:BinaryInvertedFile,Knn<:AbstractKnn}
    idx::InvFileType
    res::Knn
    n::Int
end

function Intersections.onmatch!(output::BinInvFileOutput, L, P, isize::Int)
    @inbounds objID = L[1].list[P[1]]
    @inbounds d = set_distance_evaluate(output.idx.dist, isize, output.n, output.idx.sizes[objID])
    push_item!(output.res, objID, d)
end

"""
  search_invfile(accept_posting_list::Function, idx::BinaryInvertedFile, ctx::InvertedFileContext, Q, res::AbstractKnn, t)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments

- `accept_posting_list`: predicate to accept or reject a posting list
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`select_posting_lists`](@ref)
- `t`: threshold (t=1 union, t > 1 solves the t-threshold problem)
"""
function search_invfile(idx::BinaryInvertedFile, ctx::InvertedFileContext, Q::Vector{PostType}, res::AbstractKnn, t) where {PostType<:PostingList}
    n = length(Q)
    P = getpositions(n, ctx)
    cost = xmerge!(BinInvFileOutput(idx, res, n), Q, P; t)
    add_block_evaluations!(ctx, length(Q))
    add_distance_evaluations!(ctx, cost)
    res
end
