# This file is a part of SimilaritySearch.jl

export DistantHyperplanes

"""
    DistantHyperplanes(dist::SemiMetric, X::AbstractDatabase, nbits::Int;
        hsel::Int=nbits*1024, entquantile::Float64=0.9, henc::Int=2^13,
        minbatch::Int=4, verbose::Bool=true)

Builds a hyperplane-based binary sketch generator for a generic metric space. A hyperplane
is a pair of anchor objects `(i, j)` sampled from `X`; an object `obj` falls on one side of
hyperplane `(i, j)` when `evaluate(dist, obj, X[i]) <= evaluate(dist, obj, X[j])`.

`hsel` candidate hyperplanes are sampled and characterized by which side a random subsample
of `henc` objects from `X` falls on; hyperplanes whose resulting bit-vector is not close to
a fair coin (entropy, out of a maximum of `1.0`) are discarded as uninformative. The entropy
cutoff is the `entquantile` quantile of the entropies the candidates actually achieved, i.e.
the top `1 - entquantile` fraction of candidates (by entropy) survive -- this adapts
automatically to whatever entropy ceiling the dataset/distance can actually deliver, instead
of comparing against a fixed absolute bar that could sit above that ceiling and silently
discard every candidate, producing a useless 0-bit sketch. Only a fully degenerate case --
every candidate has exactly zero entropy -- raises an error, since no cutoff can rescue it.
From the survivors, `nbits` hyperplanes are finally kept, chosen to be as mutually diverse
as possible -- i.e., spread apart in Hamming distance between their characterization
bit-vectors, up to a global bit-flip (two hyperplanes whose sides happen to be labeled
oppositely are just as good a pair to keep as two that agree) -- via a farthest-first
traversal ([`fft`](@ref)).

Use [`bitsketch`](@ref) to encode objects with the resulting `DistantHyperplanes`; the
sketches it produces are compared with [`distance`](@ref)`(m)` (Hamming distance over
`UInt64`-packed bits).

# Arguments
- `dist`: the distance function of the underlying metric space
- `X`: the database used both to sample candidate hyperplane anchors and, later, as the
  reference set hyperplanes are evaluated against when encoding new objects
- `nbits`: the number of output bits (i.e., hyperplanes) to keep; must be a multiple of 64

# Keyword Arguments
- `hsel`: number of candidate hyperplanes (pairs of objects) to sample and characterize
- `entquantile`: quantile (in `[0, 1]`) of the candidates' achieved entropies used as the
  acceptance cutoff (see above)
- `henc`: sample size used to characterize each candidate hyperplane; must be a multiple
  of 64 and smaller than `length(X)`
- `minbatch`: minimum number of items processed per parallel task (see `@BATCHES`)
- `verbose`: whether the per-center progress message of the underlying [`fft`](@ref) call
  is produced

# Examples

```julia
julia> using SimilaritySearch

julia> X = MatrixDatabase(rand(Float32, 8, 10_000));

julia> m = SimilaritySearch.Projections.DistantHyperplanes(SimilaritySearch.Dist.L2(), X, 128);

julia> B = SimilaritySearch.Projections.bitsketch(m, X);

julia> size(B.matrix), eltype(B.matrix)  # (2, 10000), UInt64 -- 128 bits / 64 = 2 words per sketch
```
"""
struct DistantHyperplanes{D<:SemiMetric,DB<:AbstractDatabase}
    dist::D
    H::Vector{Pair{Int,Int}}
    C::DB
end

distance(::DistantHyperplanes) = Hamming()

"""
    outdim(m::DistantHyperplanes)

Returns the number of output bits (hyperplanes) of `m`.
"""
outdim(m::DistantHyperplanes) = length(m.H)

function _dh_sample_pairs(n::Int, k::Int)
    visited = Set{Pair{Int,Int}}()
    P = Pair{Int,Int}[]
    sizehint!(P, k)

    for _ in 1:k
        i = rand(1:n-1)
        j = rand(i+1:n)
        p = i => j
        if p ∉ visited
            push!(visited, p)
            push!(P, p)
        end
    end

    P
end

@inline _dh_side(dist::SemiMetric, X::AbstractDatabase, h::Pair, obj) =
    evaluate(dist, obj, X[h[1]]) <= evaluate(dist, obj, X[h[2]])

function _dh_entropy(binvec::AbstractVector{UInt64})
    n = length(binvec) * 64
    c1 = sum(count_ones, binvec)
    c0 = n - c1
    (c0 == 0 || c1 == 0) && return 0.0  # avoid 0 * log2(0) == NaN
    p0 = c0 / n
    p1 = c1 / n
    -p0 * log2(p0) - p1 * log2(p1)
end

"""
    _dh_entropy_threshold(ents::Vector{Float64}, entquantile::Float64, verbose::Bool)

Picks the entropy cutoff used to discard uninformative candidate hyperplanes: the
`entquantile` quantile of the entropies actually achieved by the candidates (`ents`). This
adapts automatically to whatever entropy ceiling the dataset/distance can actually deliver
(see issue #55), instead of comparing against a fixed absolute bar that could silently
discard every candidate. Throws when every candidate is completely uninformative (entropy
`0`), since no cutoff can rescue that case.
"""
function _dh_entropy_threshold(ents::AbstractVector{Float64}, entquantile::Float64, verbose::Bool)
    maximum(ents) > 0 || throw(ArgumentError("none of the $(length(ents)) candidate hyperplanes carry any information (all have zero characterization entropy); cannot build a sketch for this dataset/distance"))
    thresh = quantile(ents, entquantile)
    verbose && @info "hyperplane characterization: entropy threshold=$(round(thresh; digits=4)) (entquantile=$entquantile), mean entropy=$(round(mean(ents); digits=4)), max entropy=$(round(maximum(ents); digits=4))"
    thresh
end

"""
    DualHammingDistance <: SemiMetric

Hamming distance between hyperplane-characterization bit-vectors, made symmetric under a
global bit-flip: which side of a hyperplane gets labeled `0` and which `1` is arbitrary
(swapping a pair's two anchors flips every bit), so two candidates whose bit-vectors are
exact complements of each other are just as redundant as two whose bit-vectors match
exactly. Used internally to select a mutually diverse subset of hyperplanes via [`fft`](@ref).
"""
struct DualHammingDistance <: SemiMetric end

function evaluate(::DualHammingDistance, u, v)
    d = 0
    @inbounds for i in eachindex(u)
        d += count_ones(u[i] ⊻ v[i])
    end

    n = 64 * length(u)
    min(d, n - d)
end

function DistantHyperplanes(dist::SemiMetric, X::AbstractDatabase, nbits::Int;
        hsel::Int=nbits * 1024,     # number of candidate hyperplanes to evaluate
        entquantile::Float64=0.9,   # quantile of achieved entropy used as acceptance cutoff
        henc::Int=2^13,             # characterizes hyperplanes with this many objects
        minbatch::Int=4,
        verbose::Bool=true)
    nbits % 64 == 0 || throw(ArgumentError("nbits should be a factor of 64"))
    nbits <= hsel || throw(ArgumentError("hsel should be bigger than nbits"))
    henc % 64 == 0 || throw(ArgumentError("henc should be a factor of 64"))
    length(X) > henc || throw(ArgumentError("henc ($henc) should be smaller than |X| ($(length(X)))"))

    P = _dh_sample_pairs(length(X), hsel)
    S = shuffle!(collect(1:length(X)))
    resize!(S, henc)
    sort!(S)
    B = BitArray(undef, henc, length(P))
    objs = [X[S[i]] for i in 1:henc]

    # parallelized over candidates (columns), not sample objects (rows): each column owns a
    # whole, exclusive set of `BitArray` words (henc is a multiple of 64), so this is
    # race-free regardless of how @BATCHES chunks the column range -- see issue #57.
    @BATCHES minbatch for j in eachindex(P)
        p = P[j]
        for i in 1:henc
            B[i, j] = _dh_side(dist, X, p, objs[i])
        end
    end

    H = reshape(B.chunks, (henc ÷ 64, length(P))) |> MatrixDatabase
    ents = [_dh_entropy(H[i]) for i in eachindex(H)]
    thresh = _dh_entropy_threshold(ents, entquantile, verbose)
    E = ents .>= thresh
    Psel, Hsel = P[E], H.matrix[:, E] |> MatrixDatabase

    F = fft(DualHammingDistance(), Hsel, nbits; verbose)
    DistantHyperplanes(dist, Psel[F.centers], X)
end

"""
    bitsketch(m::DistantHyperplanes, obj) -> Vector{UInt64}
    bitsketch(m::DistantHyperplanes, X::AbstractDatabase; minbatch::Int=4) -> MatrixDatabase

Encodes `obj` (or every object of `X`) with the hyperplanes of `m`: bit `i` is `1` when
`obj` falls on the "`<=`" side of hyperplane `m.H[i]` (see [`DistantHyperplanes`](@ref)),
packed into `UInt64` words. Sketches are compared with [`distance`](@ref)`(m)`.

# Arguments
- `m`: the fitted `DistantHyperplanes` sketch generator
- `obj`/`X`: the object, or database of objects, to sketch
- `minbatch`: (database method only) minimum number of items processed per parallel task
"""
function bitsketch(m::DistantHyperplanes, obj)
    b = BitArray(undef, length(m.H))
    for i in eachindex(m.H)
        b[i] = _dh_side(m.dist, m.C, m.H[i], obj)
    end

    b.chunks
end

function bitsketch(m::DistantHyperplanes, X::AbstractDatabase; minbatch::Int=4)
    n = length(m.H)
    b = BitArray(undef, n, length(X))

    @BATCHES minbatch for j in 1:length(X)
        obj = X[j]
        for i in eachindex(m.H)
            b[i, j] = _dh_side(m.dist, m.C, m.H[i], obj)
        end
    end

    MatrixDatabase(reshape(b.chunks, (n ÷ 64, length(X))))
end
