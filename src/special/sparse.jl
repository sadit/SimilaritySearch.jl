module Sparse

using ...SimilaritySearch: Dist, AbstractDatabase
using ...SimilaritySearch.Intersections: doublingsearch
using ...SimilaritySearch.Dist: NormCosine, Cosine, NormAngle, Angle, fastacos, evaluate
import ...SimilaritySearch.Dist.CastF32: dot32, norm32
using SparseArrays
using LinearAlgebra

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
    SparseVecView(nzind, nzval)

A read-only view of a single sparse vector, given as parallel arrays of non-zero
indices `nzind` and non-zero values `nzval` (as produced by, e.g., `rowvals`/`nonzeros` on a
column of a `SparseMatrixCSC`).
"""
struct SparseVecView{IType,VType}
    nzind::IType
    nzval::VType
end

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
    SparseVecView(view(rows, r), view(vals, r))
end

end