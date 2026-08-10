module Sparse

using ...SimilaritySearch: Dist, AbstractDatabase
using ...SimilaritySearch.Intersections: doublingsearch
using ...SimilaritySearch.Dist: NormCosine, Cosine, NormAngle, Angle, fastacos, evaluate
import ...SimilaritySearch.Dist.CastF32: dot32, norm32
using SparseArrays
using LinearAlgebra

export centroid, sparsedot

"""
    SparseDatabase(M::MType) where {MType<:SparseMatrixCSC}

An `AbstractDatabase` wrapping a sparse matrix `M` (in CSC format); each column of `M`
is treated as a stored vector, and indexing the database (`db[i]`) returns the `i`-th
column as a [`SparseVecView`](@ref).
"""
struct SparseDatabase{MType<:SparseMatrixCSC} <: AbstractDatabase
    M::MType
end

"""
    SparseVecView(n, nzind, nzval)

A read-only view of a single sparse vector of length `n`, given as parallel arrays of non-zero
indices `nzind` and non-zero values `nzval` (as produced by, e.g., `rowvals`/`nonzeros` on a
column of a `SparseMatrixCSC`).
"""
struct SparseVecView{IType,VType}
    n::Int
    nzind::IType
    nzval::VType
end

Base.length(v::SparseVecView) = v.n
SparseArrays.nnz(v::SparseVecView) = length(v.nzind)

const SparseVectorLike = Union{SparseVector, SparseVecView}

function _dot_linear(ai::AbstractVector{Ti}, av::AbstractVector{Tv}, bi::AbstractVector{Ti}, bv::AbstractVector{Tv}) where {Ti,Tv}
    na, nb = length(ai), length(bi)
    s = zero(Tv)
    i = j = 1
    @inbounds while i <= na && j <= nb
        xa, xb = ai[i], bi[j]
        if xa == xb
            s += av[i] * bv[j]
            i += 1; j += 1
        elseif xa < xb
            i += 1
        else
            j += 1
        end
    end

    s
end

function _dot_gallop(ai::AbstractVector{Ti}, av::AbstractVector{Tv}, bi::AbstractVector{Ti}, bv::AbstractVector{Tv}) where {Ti,Tv}
    # ai/av MUST be the smaller side
    na, nb = length(ai), length(bi)
    s = zero(Tv)
    pos = 1
    @inbounds for i in 1:na
        pos > nb && break
        x = ai[i]
        pos = doublingsearch(bi, x, pos, nb)
        if pos <= nb && bi[pos] == x
            s += av[i] * bv[pos]
        end
    end

    s
end

"""
    sparsedot(a, b; small_threshold::Int=30, ratio_threshold::Float32=3.0f0)

Adaptive dot product between two `SparseVector`s or `SparseVecView`s:
- both sides have fewer than `small_threshold` stored entries, or their sizes are
  within `ratio_threshold` of each other: a plain linear merge.
- otherwise (one side much larger than the other): a Hwang-Lin/galloping merge.
"""
function sparsedot(a::SparseVectorLike, b::SparseVectorLike;
        small_threshold::Int=30, ratio_threshold::Real=3.0f0)
    ai, av, bi, bv = a.nzind, a.nzval, b.nzind, b.nzval
    na, nb = length(ai), length(bi)
    if na == 0 || nb == 0
        return 0f0
    end
    lo, hi = na <= nb ? (na, nb) : (nb, na)
    rt = Float32(ratio_threshold)
    if hi < small_threshold || hi / lo <= rt
        return Float32(_dot_linear(ai, av, bi, bv))
    end

    Float32(na <= nb ? _dot_gallop(ai, av, bi, bv) : _dot_gallop(bi, bv, ai, av))
end

function dot32(a::SparseVectorLike, b::SparseVectorLike)
    sparsedot(a, b)
end

function dot32(A::SparseVectorLike, B)
    s = 0f0
    @inbounds for (i, j) in enumerate(A.nzind)
        s += Float32(A.nzval[i]) * Float32(B[j])
    end

    s
end

dot32(B, A::SparseVectorLike) = dot32(A, B)

function norm32(A::SparseVectorLike)
    sqrt(dot32(A, A))
end

Dist.evaluate(::NormCosine, A::SparseVectorLike, B::SparseVectorLike) = 1f0 - dot32(A, B)
Dist.evaluate(::Cosine, A::SparseVectorLike, B::SparseVectorLike) = 1f0 - dot32(A, B) / (norm32(A) * norm32(B))
Dist.evaluate(::NormAngle, A::SparseVectorLike, B::SparseVectorLike) = fastacos(dot32(A, B))
Dist.evaluate(::Angle, A::SparseVectorLike, B::SparseVectorLike) = fastacos(dot32(A, B) / (norm32(A) * norm32(B)))

Base.length(D::SparseDatabase) = size(D.M, 2)
LinearAlgebra.dot(A::SparseVecView, B::SparseVecView) = dot32(A, B)
LinearAlgebra.dot(A::SparseVecView, B) = dot32(A, B)
LinearAlgebra.dot(A, B::SparseVecView) = dot32(B, A)
LinearAlgebra.norm(A::SparseVecView) = norm32(A)

function LinearAlgebra.normalize!(A::SparseVecView)
    inv = 1f0/norm32(A)
    for i in eachindex(A.nzval)
        A.nzval[i] *= inv
    end
    A
end

function Base.getindex(D::SparseDatabase, i)
    r = nzrange(D.M, i)
    rows = rowvals(D.M)
    vals = nonzeros(D.M)
    SparseVecView(size(D.M, 1), view(rows, r), view(vals, r))
end

"""
    Base.sum(cluster::AbstractVector{<:SparseVectorLike})

`SparseVector` counterpart of `sum(::AbstractVector{<:Dict})`: concatenate every
`(index, value)` pair from every input vector, sort once by index, then combine
consecutive equal indices in a single linear pass. Beat every alternative tried
(naive `+`-folding, pairwise tree merging, a k-way heap merge) at every cluster size
benchmarked, and — unlike a dense-accumulator approach — its cost does not depend on
the vectors' dimension, only on their total number of stored entries. See
[sadit/TextSearch.jl#25](https://github.com/sadit/TextSearch.jl/issues/25).

All vectors in `cluster` must have the same dimension (`length`).
"""
function Base.sum(cluster::AbstractVector{<:SparseVectorLike})
    n = length(cluster[1])
    total = sum(nnz, cluster)
    Tv = eltype(cluster[1].nzval)
    Ti = eltype(cluster[1].nzind)
    
    all_ind = Vector{Ti}(undef, total)
    all_val = Vector{Tv}(undef, total)
    p = 1
    for v in cluster
        @assert length(v) == n "all vectors in `cluster` must share the same dimension"
        m = nnz(v)
        copyto!(all_ind, p, v.nzind, 1, m)
        copyto!(all_val, p, v.nzval, 1, m)
        p += m
    end

    perm = sortperm(all_ind)
    permute!(all_ind, perm)
    permute!(all_val, perm)

    out_ind = Vector{Ti}()
    sizehint!(out_ind, total)
    out_val = Vector{Tv}()
    sizehint!(out_val, total)
    
    i = 1
    @inbounds while i <= total
        j = i
        s = all_val[i]
        while j < total && all_ind[j+1] == all_ind[i]
            j += 1
            s += all_val[j]
        end
        push!(out_ind, all_ind[i])
        push!(out_val, s)
        i = j + 1
    end

    SparseVector(n, out_ind, out_val)
end

"""
    centroid(cluster::AbstractVector{<:SparseVectorLike})

Centroid (normalized sum) of a cluster of `SparseVector`s. See
[`sum(::AbstractVector{<:SparseVectorLike})`](@ref sum) for the algorithm.
"""
centroid(cluster::AbstractVector{<:SparseVectorLike}) = normalize!(sum(cluster))

end