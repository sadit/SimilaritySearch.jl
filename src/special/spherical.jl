# This file is a part of SimilaritySearch.jl

"""
    Spherical

Implements the spherical embedding of Neyshabur & Srebro, "On Symmetric and Asymmetric
LSHs for Inner Product Search" (2015): a dataset-dependent transform that turns Maximum
Inner Product Search (MIPS) into ordinary nearest-neighbor search under a standard metric
(e.g. squared Euclidean or `NormCosine`).

The scheme is *asymmetric*: the dataset and the queries are mapped with two different
functions, [`transform`](@ref)/[`transform!`](@ref) (data-side, `P`) and
[`transform_query`](@ref)/[`transform_query!`](@ref) (query-side, `Q`):

```math
P(x) = \\left[\\frac{x}{M}, \\sqrt{1 - \\left\\|\\frac{x}{M}\\right\\|^2}\\right]
\\qquad
Q(q) = \\left[\\frac{q}{\\|q\\|}, 0\\right]
```

where ``M`` is the maximum norm over the fitted dataset. Both `P(x)` and `Q(q)` land
exactly on the unit sphere, and for any fixed query, ranking data points by increasing
`||P(x) - Q(q)||` (or decreasing dot product) recovers exactly the ranking by decreasing
inner product ``x \\cdot q`` -- `M` and `\\|q\\|` are constants w.r.t. `x`, so they do not
affect the ordering. See [`SphericalEmbedding`](@ref) for how `M` is fitted and stored, and
its docstring for the caveat that matters when the underlying database keeps growing.

Both dense (`AbstractVector`/`AbstractMatrix`/`MatrixDatabase`) and sparse
(`Special.Sparse.SparseVecView`/`SparseDatabase`, and plain `SparseArrays.SparseVector`/
`SparseMatrixCSC`) representations are supported.
"""
module Spherical

export SphericalEmbedding, indim, outdim, transform, transform!, transform_query, transform_query!

using SparseArrays
using ...SimilaritySearch: AbstractDatabase, MatrixDatabase, getminbatch,
    @BATCHES, @BEGIN, @BEGINBATCH, @LOOP, @ENDBATCH, @END, @nbatches, @batchid
import ...SimilaritySearch.Dist.CastF32: norm32
using ..Sparse: SparseVecView, SparseDatabase

"""
    SphericalEmbedding(X::AbstractMatrix; pad::Bool=true, padmultiple::Int=8, maxnorm=nothing)
    SphericalEmbedding(db::MatrixDatabase; pad::Bool=true, padmultiple::Int=8, maxnorm=nothing)
    SphericalEmbedding(X::SparseMatrixCSC; maxnorm=nothing)
    SphericalEmbedding(db::Special.Sparse.SparseDatabase; maxnorm=nothing)

Fits a spherical embedding (Neyshabur & Srebro) over a dataset, storing the metadata
needed by [`transform`](@ref)/[`transform!`](@ref)/[`transform_query`](@ref)/
[`transform_query!`](@ref): the dataset's maximum norm `maxnorm` (`M`), its input
dimension `indim`, and (dense only) an optional number of extra zero-padding coordinates
`pad`. The embedded (output) dimension is `indim + pad + 1` (the `+1` is the residual-norm
coordinate appended by [`transform`](@ref)); see [`outdim`](@ref).

Since `maxnorm` is computed once, from the dataset given here, this struct is exactly the
"fitted state" that must be saved and reused: apply the SAME `SphericalEmbedding` to the
dataset (via [`transform`](@ref)) and to every later query (via
[`transform_query`](@ref)) -- never re-fit a new one per query.

!!! warning "Growing databases"
    `maxnorm` is a property of the dataset **at fit time**. If the database keeps growing
    and a later vector's norm exceeds `maxnorm`, this `SphericalEmbedding` is stale: see
    the warning on [`transform!`](@ref) for what happens and what to do about it (refit a
    new `SphericalEmbedding`, recomputing `maxnorm` over the enlarged dataset).

# Arguments
- `X`/`db`: the dataset to fit against; dense forms accept an `AbstractMatrix` (columns
  are objects) or a [`MatrixDatabase`](@ref); sparse forms accept a `SparseMatrixCSC`
  (columns are objects) or a [`Special.Sparse.SparseDatabase`](@ref).

# Keyword Arguments
- `pad`: dense only. When `true` (the default), pads the embedded dimension up to the
  next multiple of `padmultiple` with extra zero coordinates -- zero-valued coordinates do
  not change any dot product/norm, so this is purely a memory-layout knob (e.g. for
  SIMD-friendly alignment), never a correctness one. Sparse inputs do not support padding
  (there is nothing to align; padding a sparse vector would just mean storing explicit
  zeros).
- `padmultiple`: dense only, the multiple to pad `indim + 1` up to when `pad=true`
  (default `8`).
- `maxnorm`: an optional precomputed maximum norm to use instead of scanning `X`/`db`
  (e.g. to reuse a previously-fitted value, or to deliberately fit a looser bound that
  tolerates some future growth without going stale).

# Examples

```julia
julia> using SimilaritySearch, SimilaritySearch.Special.Spherical

julia> X = rand(Float32, 8, 1000);

julia> se = SphericalEmbedding(X);

julia> outdim(se)  # 8 + pad + 1
16

julia> P = transform(se, X);       # embed the dataset

julia> q = rand(Float32, 8);

julia> Qq = transform_query(se, q); # embed a query the SAME way

julia> length(Qq) == outdim(se)
true
```
"""
struct SphericalEmbedding
    maxnorm::Float32
    indim::Int
    pad::Int

    function SphericalEmbedding(maxnorm::Real, indim::Integer, pad::Integer)
        maxnorm > 0 || throw(ArgumentError("SphericalEmbedding: maxnorm=$maxnorm must be positive (an all-zero dataset has no well-defined spherical embedding)"))
        new(Float32(maxnorm), Int(indim), Int(pad))
    end
end

"Rounds `n` up to the next multiple of `padmultiple` (or `0` extra if already a multiple, or if `padmultiple <= 1`)."
function _paddingfor(n::Integer, padmultiple::Integer)
    padmultiple <= 1 && return 0
    r = n % padmultiple
    r == 0 ? 0 : padmultiple - r
end

function _maxnorm_dense(X::AbstractMatrix)
    n = size(X, 2)
    minbatch = getminbatch(n)
    local best

    @BATCHES minbatch begin
    @BEGIN
        B = zeros(Float32, @nbatches())
    @BEGINBATCH
        b = 0f0
    @LOOP for j in 1:n
        b = max(b, norm32(view(X, :, j)))
    end
    @ENDBATCH
        B[@batchid()] = b
    @END
        best = maximum(B)
    end

    best
end

function _maxnorm_sparsedb(db::SparseDatabase)
    best = 0f0
    for i in eachindex(db)
        best = max(best, norm32(db[i]))
    end

    best
end

function _maxnorm_sparsematrix(X::SparseMatrixCSC)
    vals = nonzeros(X)
    best = 0f0
    for j in 1:size(X, 2)
        s = 0f0
        for k in nzrange(X, j)
            v = Float32(vals[k])
            s = muladd(v, v, s)
        end
        best = max(best, sqrt(s))
    end

    best
end

function SphericalEmbedding(X::AbstractMatrix; pad::Bool=true, padmultiple::Int=8, maxnorm=nothing)
    d = size(X, 1)
    M = maxnorm === nothing ? _maxnorm_dense(X) : Float32(maxnorm)
    p = pad ? _paddingfor(d + 1, padmultiple) : 0
    SphericalEmbedding(M, d, p)
end

SphericalEmbedding(db::MatrixDatabase; kwargs...) = SphericalEmbedding(db.matrix; kwargs...)

function SphericalEmbedding(db::SparseDatabase; maxnorm=nothing)
    M = maxnorm === nothing ? _maxnorm_sparsedb(db) : Float32(maxnorm)
    SphericalEmbedding(M, size(db.M, 1), 0)
end

function SphericalEmbedding(X::SparseMatrixCSC; maxnorm=nothing)
    M = maxnorm === nothing ? _maxnorm_sparsematrix(X) : Float32(maxnorm)
    SphericalEmbedding(M, size(X, 1), 0)
end

"""
    indim(se::SphericalEmbedding) -> Int

Input dimension expected by [`transform`](@ref)/[`transform_query`](@ref): the dimension
of the original vectors, not counting any padding or the appended residual coordinate.
"""
indim(se::SphericalEmbedding) = se.indim

"""
    outdim(se::SphericalEmbedding) -> Int

Output (embedded) dimension produced by [`transform`](@ref)/[`transform_query`](@ref):
`indim(se) + pad + 1` (the `+1` is the appended coordinate -- the residual norm for
[`transform`](@ref), always `0` for [`transform_query`](@ref)).
"""
outdim(se::SphericalEmbedding) = se.indim + se.pad + 1

"""
    transform!(se::SphericalEmbedding, out::AbstractVector, x::AbstractVector) -> out

In-place data-side spherical embedding (`P(x)`, see [`Spherical`](@ref)): scales `x` by
`1/se.maxnorm`, fills any padding coordinates with zero, and appends the residual-norm
coordinate `sqrt(1 - ||x/maxnorm||^2)`, so that `out` lands exactly on the unit sphere.
`out` must have length `outdim(se)`.

!!! warning "Stale `maxnorm`"
    If `norm(x) > se.maxnorm` -- e.g. `x` is a new vector inserted into a database that
    kept growing after `se` was fitted -- the residual term's argument goes negative; it
    is clamped to `0` here rather than throwing a `DomainError`, but the ranking
    guarantee described in [`Spherical`](@ref) no longer holds for `x` (and for every
    comparison against it) once that happens. When the dataset can keep growing, either
    refit a new `SphericalEmbedding` (recomputing `maxnorm` over the enlarged dataset) or
    fit the original one with a deliberately loose `maxnorm=` bound that tolerates the
    growth you expect.
"""
function transform!(se::SphericalEmbedding, out::AbstractVector, x::AbstractVector)
    length(x) == se.indim || throw(DimensionMismatch("SphericalEmbedding.transform!: length(x)=$(length(x)) must equal indim(se)=$(se.indim)"))
    length(out) == outdim(se) || throw(DimensionMismatch("SphericalEmbedding.transform!: length(out)=$(length(out)) must equal outdim(se)=$(outdim(se))"))

    invM = 1f0 / se.maxnorm
    s = 0f0
    @inbounds for i in 1:se.indim
        v = Float32(x[i]) * invM
        out[i] = v
        s = muladd(v, v, s)
    end
    @inbounds for i in se.indim+1:se.indim+se.pad
        out[i] = 0f0
    end
    @inbounds out[end] = sqrt(max(1f0 - s, 0f0))
    out
end

"""
    transform(se::SphericalEmbedding, x::AbstractVector) -> Vector{Float32}

Out-of-place version of [`transform!`](@ref): returns a freshly allocated
`Vector{Float32}` of length `outdim(se)`.
"""
function transform(se::SphericalEmbedding, x::AbstractVector)
    out = Vector{Float32}(undef, outdim(se))
    transform!(se, out, x)
end

"""
    transform!(se::SphericalEmbedding, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4) -> O

In-place version of `transform(se, X)`: embeds every column of `X` (see [`transform!`](@ref))
into the corresponding column of `O`, which must have `outdim(se)` rows and the same
number of columns as `X`. Columns are embedded in parallel using `@BATCHES`.
"""
function transform!(se::SphericalEmbedding, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)

    @BATCHES minbatch for j in 1:n
        transform!(se, view(O, :, j), view(X, :, j))
    end

    O
end

"""
    transform(se::SphericalEmbedding, X::AbstractMatrix; minbatch::Int=4) -> Matrix{Float32}

Embeds every column (object) of `X` using [`transform`](@ref), returning a new
`Matrix{Float32}` with `outdim(se)` rows and the same number of columns as `X`.
"""
function transform(se::SphericalEmbedding, X::AbstractMatrix; minbatch::Int=4)
    O = Matrix{Float32}(undef, outdim(se), size(X, 2))
    transform!(se, O, X; minbatch)
end

"""
    transform(se::SphericalEmbedding, db::MatrixDatabase; minbatch::Int=4) -> MatrixDatabase

Embeds every object of `db` using [`transform`](@ref), returning a new `MatrixDatabase`
wrapping a freshly allocated `Matrix{Float32}`.
"""
transform(se::SphericalEmbedding, db::MatrixDatabase; kwargs...) = MatrixDatabase(transform(se, db.matrix; kwargs...))

"""
    transform(se::SphericalEmbedding, x::Special.Sparse.SparseVecView) -> SparseVecView

Sparse-vector version of [`transform`](@ref): scales the stored nonzero entries of `x` by
`1/se.maxnorm` and appends one explicit `(outdim(se), residual)` entry, reusing
[`Special.Sparse`](@ref)'s existing distance machinery unchanged (the appended index is
always the largest, so the result stays sorted). `se` must have been fitted without
padding (`se.pad == 0`, the default for sparse inputs).
"""
function transform(se::SphericalEmbedding, x::SparseVecView)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform: sparse inputs require pad=0 (got pad=$(se.pad))"))
    invM = 1f0 / se.maxnorm
    m = length(x.nzind)
    I = Vector{eltype(x.nzind)}(undef, m + 1)
    V = Vector{Float32}(undef, m + 1)
    s = 0f0
    @inbounds for i in 1:m
        v = Float32(x.nzval[i]) * invM
        I[i] = x.nzind[i]
        V[i] = v
        s = muladd(v, v, s)
    end
    I[end] = se.indim + 1
    V[end] = sqrt(max(1f0 - s, 0f0))
    SparseVecView(outdim(se), I, V)
end

"""
    transform(se::SphericalEmbedding, x::SparseArrays.SparseVector) -> SparseArrays.SparseVector

Version of [`transform`](@ref) for a plain `SparseArrays.SparseVector` (as opposed to
[`Special.Sparse.SparseVecView`](@ref)); same semantics.
"""
function transform(se::SphericalEmbedding, x::SparseVector)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform: sparse inputs require pad=0 (got pad=$(se.pad))"))
    invM = 1f0 / se.maxnorm
    nzind = SparseArrays.nonzeroinds(x)
    nzval = nonzeros(x)
    m = length(nzval)
    I = Vector{Int}(undef, m + 1)
    V = Vector{Float32}(undef, m + 1)
    s = 0f0
    @inbounds for i in 1:m
        v = Float32(nzval[i]) * invM
        I[i] = nzind[i]
        V[i] = v
        s = muladd(v, v, s)
    end
    I[end] = se.indim + 1
    V[end] = sqrt(max(1f0 - s, 0f0))
    sparsevec(I, V, outdim(se))
end

"""
    transform(se::SphericalEmbedding, X::SparseMatrixCSC) -> SparseMatrixCSC

Embeds every column of the sparse matrix `X` (see [`transform`](@ref)), returning a new
`SparseMatrixCSC` with `outdim(se)` rows and the same number of columns as `X`.
"""
function transform(se::SphericalEmbedding, X::SparseMatrixCSC)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform: sparse inputs require pad=0 (got pad=$(se.pad))"))
    invM = 1f0 / se.maxnorm
    n = size(X, 2)
    nz = nnz(X)
    rows = rowvals(X)
    vals = nonzeros(X)
    I = Vector{Int}(undef, nz + n)
    J = Vector{Int}(undef, nz + n)
    V = Vector{Float32}(undef, nz + n)

    p = 1
    @inbounds for j in 1:n
        s = 0f0
        for k in nzrange(X, j)
            v = Float32(vals[k]) * invM
            I[p] = rows[k]
            J[p] = j
            V[p] = v
            s = muladd(v, v, s)
            p += 1
        end
        I[p] = se.indim + 1
        J[p] = j
        V[p] = sqrt(max(1f0 - s, 0f0))
        p += 1
    end

    sparse(I, J, V, outdim(se), n)
end

"""
    transform(se::SphericalEmbedding, db::Special.Sparse.SparseDatabase) -> SparseDatabase

Embeds every object of `db` using [`transform`](@ref), returning a new `SparseDatabase`.
"""
transform(se::SphericalEmbedding, db::SparseDatabase) = SparseDatabase(transform(se, db.M))

"""
    transform_query!(se::SphericalEmbedding, out::AbstractVector, q::AbstractVector) -> out

In-place query-side spherical embedding (`Q(q)`, see [`Spherical`](@ref)): unlike
[`transform!`](@ref), `q` is scaled by its OWN norm (not `se.maxnorm`), and every padding
coordinate together with the final appended coordinate is set to `0` (there is no residual
term to compute on the query side -- it is always exactly `0`, by construction). `out`
must have length `outdim(se)`. A zero `q` maps to an all-zero `out`.

Must be applied with the SAME `se` used to [`transform`](@ref) the dataset being searched
-- `se` only contributes `indim`/`outdim` bookkeeping here (`maxnorm` is not used at all),
so, unlike [`transform!`](@ref), this function itself never goes stale as the dataset
grows; only re-fitting `se` (which changes `outdim`) would require re-embedding queries.
"""
function transform_query!(se::SphericalEmbedding, out::AbstractVector, q::AbstractVector)
    length(q) == se.indim || throw(DimensionMismatch("SphericalEmbedding.transform_query!: length(q)=$(length(q)) must equal indim(se)=$(se.indim)"))
    length(out) == outdim(se) || throw(DimensionMismatch("SphericalEmbedding.transform_query!: length(out)=$(length(out)) must equal outdim(se)=$(outdim(se))"))

    nq = norm32(q)
    invn = nq > 0 ? 1f0 / nq : 0f0
    @inbounds for i in 1:se.indim
        out[i] = Float32(q[i]) * invn
    end
    @inbounds for i in se.indim+1:outdim(se)
        out[i] = 0f0
    end
    out
end

"""
    transform_query(se::SphericalEmbedding, q::AbstractVector) -> Vector{Float32}

Out-of-place version of [`transform_query!`](@ref).
"""
function transform_query(se::SphericalEmbedding, q::AbstractVector)
    out = Vector{Float32}(undef, outdim(se))
    transform_query!(se, out, q)
end

"""
    transform_query!(se::SphericalEmbedding, O::AbstractMatrix, Q::AbstractMatrix; minbatch::Int=4) -> O

In-place version of `transform_query(se, Q)`.
"""
function transform_query!(se::SphericalEmbedding, O::AbstractMatrix, Q::AbstractMatrix; minbatch::Int=4)
    n = size(Q, 2)

    @BATCHES minbatch for j in 1:n
        transform_query!(se, view(O, :, j), view(Q, :, j))
    end

    O
end

"""
    transform_query(se::SphericalEmbedding, Q::AbstractMatrix; minbatch::Int=4) -> Matrix{Float32}

Embeds every column (query) of `Q` using [`transform_query`](@ref), returning a new
`Matrix{Float32}` with `outdim(se)` rows and the same number of columns as `Q`.
"""
function transform_query(se::SphericalEmbedding, Q::AbstractMatrix; minbatch::Int=4)
    O = Matrix{Float32}(undef, outdim(se), size(Q, 2))
    transform_query!(se, O, Q; minbatch)
end

transform_query(se::SphericalEmbedding, db::MatrixDatabase; kwargs...) = MatrixDatabase(transform_query(se, db.matrix; kwargs...))

"""
    transform_query(se::SphericalEmbedding, q::Special.Sparse.SparseVecView) -> SparseVecView
    transform_query(se::SphericalEmbedding, q::SparseArrays.SparseVector) -> SparseArrays.SparseVector

Sparse-vector version of [`transform_query`](@ref). Unlike [`transform`](@ref)'s sparse
methods, no extra entry is appended -- the query's appended coordinate is always exactly
`0`, and sparse formats already represent unlisted entries as `0` implicitly.
"""
function transform_query(se::SphericalEmbedding, q::SparseVecView)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform_query: sparse inputs require pad=0 (got pad=$(se.pad))"))
    nq = norm32(q)
    invn = nq > 0 ? 1f0 / nq : 0f0
    SparseVecView(outdim(se), copy(q.nzind), Float32.(q.nzval) .* invn)
end

function transform_query(se::SphericalEmbedding, q::SparseVector)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform_query: sparse inputs require pad=0 (got pad=$(se.pad))"))
    nzind = SparseArrays.nonzeroinds(q)
    nzval = nonzeros(q)
    s = 0f0
    @inbounds for v in nzval
        s = muladd(Float32(v), Float32(v), s)
    end
    nq = sqrt(s)
    invn = nq > 0 ? 1f0 / nq : 0f0
    SparseVector(outdim(se), copy(nzind), Float32.(nzval) .* invn)
end

"""
    transform_query!(se::SphericalEmbedding, q::Special.Sparse.SparseVecView) -> SparseVecView
    transform_query!(se::SphericalEmbedding, q::SparseArrays.SparseVector) -> SparseArrays.SparseVector

In-place version of [`transform_query`](@ref) for sparse vectors. This method modifies the 
input vector's values in-place (avoiding allocations for the values), but it returns a new 
vector/view object because the output dimensionality is `outdim(se)` (typically `dim + 1`).
"""
function transform_query!(se::SphericalEmbedding, q::SparseVecView)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform_query!: sparse inputs require pad=0 (got pad=$(se.pad))"))
    nq = norm32(q)
    invn = nq > 0 ? 1f0 / nq : 0f0
    @inbounds for i in eachindex(q.nzval)
        q.nzval[i] = q.nzval[i] * invn
    end
    SparseVecView(outdim(se), q.nzind, q.nzval)
end

function transform_query!(se::SphericalEmbedding, q::SparseVector)
    se.pad == 0 || throw(ArgumentError("SphericalEmbedding.transform_query!: sparse inputs require pad=0 (got pad=$(se.pad))"))
    nzind = SparseArrays.nonzeroinds(q)
    nzval = nonzeros(q)
    s = 0f0
    @inbounds for v in nzval
        s = muladd(Float32(v), Float32(v), s)
    end
    nq = sqrt(s)
    invn = nq > 0 ? 1f0 / nq : 0f0
    @inbounds for i in eachindex(nzval)
        nzval[i] = nzval[i] * invn
    end
    SparseVector(outdim(se), nzind, nzval)
end

end
