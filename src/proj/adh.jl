# This file is a part of SimilaritySearch.jl

export AnchoredDistantHyperplanes

"""
    AnchoredDistantHyperplanes(dist::SemiMetric, X::AbstractDatabase, nbits::Int;
        anchor=nothing, anchorpolicy::Symbol=:random,
        hsel::Int=nbits*1024, minent::Float64=0.99, henc::Int=2^13,
        minbatch::Int=4, verbose::Bool=true)

A variant of [`DistantHyperplanes`](@ref) that avoids the flip ambiguity of hyperplanes
by anchoring, instead of masking it after the fact.

A hyperplane defined by anchors `(i, j)` is, geometrically, the very same hyperplane as
`(j, i)`: relabeling which point is "first" just flips every bit of its characterization
(the side called `1` becomes the side called `0`). Plain candidate sampling has no reason
to prefer one labeling over the other, so [`DistantHyperplanes`](@ref) has to compare
candidates with a flip-invariant distance (`min(hamming(u, v), hamming(u, ~v))`) to avoid
mistaking two near-identical hyperplanes -- that only happen to disagree on which side
got called `1` -- for a genuinely diverse pair.

`AnchoredDistantHyperplanes` sidesteps this by fixing the labeling convention up front: a
single reference object, the `anchor`, orders every candidate pair `(i, j)` so that `i` is
always the one closer to the anchor (swapping them if sampling produced the opposite
order). With that shared convention in place, "closer to `i`" consistently means "closer
to the anchor-proximal point" for every hyperplane, so plain Hamming distance -- not the
flip-invariant one -- is enough to tell diverse hyperplanes from redundant ones.

Whether a given anchor choice helps or hurts is an open, dataset-dependent question --
`anchor` accepts an explicit choice (an object, or an integer id into `X`) for that reason;
when left unset, one is picked automatically per `anchorpolicy`:
- `:random` (default): a uniformly random object of `X`
- `:extremal`: the farthest object, in `dist`, from a random starting point (one step of
  [`fft`](@ref)) -- tends to sit at the periphery of `X`, which may spread
  distances-to-anchor out more than a typical (e.g. random) point would

Use [`bitsketch`](@ref) to encode objects, exactly as with [`DistantHyperplanes`](@ref);
sketches are compared with [`distance`](@ref)`(m)`.

# Arguments
- `dist`: the distance function of the underlying metric space
- `X`: the database used both to sample candidate hyperplane anchors and, later, as the
  reference set hyperplanes are evaluated against when encoding new objects
- `nbits`: the number of output bits (i.e., hyperplanes) to keep; must be a multiple of 64

# Keyword Arguments
- `anchor`: the anchor object to orient hyperplane pairs by; an integer is taken as an id
  into `X` (i.e., `X[anchor]`), anything else is used directly as the anchor object.
  `nothing` (default) computes one automatically, per `anchorpolicy`
- `anchorpolicy`: `:random` or `:extremal` (see above); only used when `anchor === nothing`
- `hsel`: number of candidate hyperplanes (pairs of objects) to sample and characterize
- `minent`: minimum accepted entropy (in `[0, 1]`) of a hyperplane's characterization
  bit-vector; hyperplanes below this threshold are discarded as uninformative
- `henc`: sample size used to characterize each candidate hyperplane; must be a multiple
  of 64 and smaller than `length(X)`
- `minbatch`: minimum number of items processed per parallel task (see `@BATCHES`)
- `verbose`: whether the per-center progress message of the underlying [`fft`](@ref) call
  is produced

# Examples

```julia
julia> using SimilaritySearch

julia> X = MatrixDatabase(rand(Float32, 8, 10_000));

julia> m = SimilaritySearch.Projections.AnchoredDistantHyperplanes(SimilaritySearch.Dist.L2(), X, 128);

julia> m2 = SimilaritySearch.Projections.AnchoredDistantHyperplanes(SimilaritySearch.Dist.L2(), X, 128; anchor=1); # X[1] as anchor

julia> B = SimilaritySearch.Projections.bitsketch(m, X);
```
"""
struct AnchoredDistantHyperplanes{D<:SemiMetric,DB<:AbstractDatabase,A}
    dist::D
    H::Vector{Pair{Int,Int}}
    C::DB
    anchor::A
end

distance(::AnchoredDistantHyperplanes) = Hamming()

"""
    outdim(m::AnchoredDistantHyperplanes)

Returns the number of output bits (hyperplanes) of `m`.
"""
outdim(m::AnchoredDistantHyperplanes) = length(m.H)

function _adh_resolve_anchor(dist::SemiMetric, X::AbstractDatabase, anchor, anchorpolicy::Symbol)
    anchor === nothing || return anchor isa Integer ? X[anchor] : anchor

    if anchorpolicy === :random
        X[rand(1:length(X))]
    elseif anchorpolicy === :extremal
        F = fft(dist, X, 2; verbose=false)
        X[F.centers[2]]
    else
        throw(ArgumentError("AnchoredDistantHyperplanes: unknown anchorpolicy :$anchorpolicy (expected :random or :extremal)"))
    end
end

function _adh_orient_pairs!(dist::SemiMetric, X::AbstractDatabase, anchor, P::Vector{Pair{Int,Int}})
    for k in eachindex(P)
        i, j = P[k]
        if evaluate(dist, anchor, X[i]) > evaluate(dist, anchor, X[j])
            P[k] = j => i
        end
    end

    P
end

function AnchoredDistantHyperplanes(dist::SemiMetric, X::AbstractDatabase, nbits::Int;
        anchor=nothing,
        anchorpolicy::Symbol=:random,
        hsel::Int=nbits * 1024,
        minent::Float64=0.99,
        henc::Int=2^13,
        minbatch::Int=4,
        verbose::Bool=true)
    nbits % 64 == 0 || throw(ArgumentError("nbits should be a factor of 64"))
    nbits <= hsel || throw(ArgumentError("hsel should be bigger than nbits"))
    henc % 64 == 0 || throw(ArgumentError("henc should be a factor of 64"))
    length(X) > henc || throw(ArgumentError("henc ($henc) should be smaller than |X| ($(length(X)))"))

    a = _adh_resolve_anchor(dist, X, anchor, anchorpolicy)

    P, H = let
        P = _dh_sample_pairs(length(X), hsel)
        _adh_orient_pairs!(dist, X, a, P)
        S = shuffle!(collect(1:length(X)))
        resize!(S, henc)
        sort!(S)
        B = BitArray(undef, henc, length(P))

        @BATCHES minbatch for i in 1:henc
            obj = X[S[i]]
            for j in eachindex(P)
                B[i, j] = _dh_side(dist, X, P[j], obj)
            end
        end

        H = reshape(B.chunks, (henc ÷ 64, length(P))) |> MatrixDatabase
        E = [_dh_entropy(H[i]) >= minent for i in eachindex(H)]
        P[E], H.matrix[:, E] |> MatrixDatabase
    end

    F = fft(Hamming(), H, nbits; verbose)
    AnchoredDistantHyperplanes(dist, P[F.centers], X, a)
end

"""
    bitsketch(m::AnchoredDistantHyperplanes, obj) -> Vector{UInt64}
    bitsketch(m::AnchoredDistantHyperplanes, X::AbstractDatabase; minbatch::Int=4) -> MatrixDatabase

Encodes `obj` (or every object of `X`) with the hyperplanes of `m`, exactly as
[`bitsketch(m::DistantHyperplanes, ...)`](@ref) does. Sketches are compared with
[`distance`](@ref)`(m)`.

# Arguments
- `m`: the fitted `AnchoredDistantHyperplanes` sketch generator
- `obj`/`X`: the object, or database of objects, to sketch
- `minbatch`: (database method only) minimum number of items processed per parallel task
"""
function bitsketch(m::AnchoredDistantHyperplanes, obj)
    b = BitArray(undef, length(m.H))
    for i in eachindex(m.H)
        b[i] = _dh_side(m.dist, m.C, m.H[i], obj)
    end

    b.chunks
end

function bitsketch(m::AnchoredDistantHyperplanes, X::AbstractDatabase; minbatch::Int=4)
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
