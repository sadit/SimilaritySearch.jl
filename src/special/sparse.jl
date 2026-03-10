module Sparse

using ...SimilaritySearch: Dist, AbstractDatabase
import ...SimilaritySearch.Dist.CastF32: dot32, norm32
using SparseArrays
using LinearAlgebra

struct SparseDatabase{MType<:SparseMatrixCSC} <: AbstractDatabase
    M::MType
end

struct SparseVecView{IType,VType}
    I::IType
    V::VType
end

struct NormCosine <: Dist.SemiMetric
end

Dist.evaluate(::NormCosine, A::SparseVecView, B::SparseVecView) = 1f0 - dot32(A, B)

function norm32(A::SparseVecView)
    sqrt(dot32(A, A))
end

function dot32(A::SparseVecView, B::SparseVecView)
    len_a::Int = length(A.I)
    len_b::Int = length(B.I)
    ia::Int = ib::Int = 1
    s = 0f0
    @inbounds while ia <= len_a && ib <= len_b
        if A.I[ia] < B.I[ib]
            ia += 1
        elseif B.I[ib] < A.I[ia]
            ib += 1
        else
            s += Float32(A.V[ia]) * Float32(B.V[ib])
            ia += 1
            ib += 1
        end
    end

    s
end

function dot32(A::SparseVecView, B)
    s = 0f0
    @inbounds for (i, j) in enumerate(A.I)
        s += Float32(A.V[i]) * Float32(B[j])
    end

    s
end

dot32(B, A::SparseVecView) = dot32(A, B)
Base.length(D::SparseDatabase) = size(D.M, 2)
LinearAlgebra.dot(A::SparseVecView, B::SparseVecView) = dot32(A, B)
LinearAlgebra.dot(A::SparseVecView, B) = dot32(A, B)
LinearAlgebra.dot(A, B::SparseVecView) = dot32(B, A)
LinearAlgebra.norm(A::SparseVecView) = norm32(A)
function LinearAlgebra.normalize!(A::SparseVecView)
    n = norm32(A)
    for i in eachindex(A.V)
        A.V[i] /= n
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