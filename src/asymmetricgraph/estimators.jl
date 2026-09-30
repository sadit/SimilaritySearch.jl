# This file is a part of SimilaritySearch.jl
#
# The estimator interface an AsymmetricSearchGraph navigates with, defined ahead of the
# submodules that implement it (ScalarQuant's SQEncoder, RaBitQ): what is stored and what a
# query becomes, the AbstractEstimator type, and the rotation an estimator may live in.

export AbstractEstimator

"""
    encode(dist::PreMetric, obj)

What an [`AsymmetricSearchGraph`](@ref) stores for the raw `obj` under `dist`: the form
`dist` evaluates a raw query against. The default returns `obj` itself, for a storage that
transforms what it stores on its own, as a [`ScalarQuant.QuantDatabase`](@ref) does under
the scalar quantizers' distances. An [`AbstractEstimator`](@ref) whose codes the storage
does not produce overrides it.
"""
encode(::PreMetric, obj) = obj

"""
    encodequery(dist::PreMetric, q)

What an [`AsymmetricSearchGraph`](@ref) evaluates `dist` with on the query side for the raw
`q`: the form `evaluate(dist, encodequery(dist, q), stored)` takes. The default returns `q`
itself. An [`AbstractEstimator`](@ref) whose evaluation needs the query prepared once --
rotated, projected, quantized on the query side -- overrides it, and the graph applies it
once per query and once per inserted item (which is the query of its own neighborhood
search), never per evaluation.
"""
encodequery(::PreMetric, q) = q

"""
    abstract type AbstractEstimator <: PreMetric end

A distance that is an estimator: it evaluates a raw query against an *encoded* object, and
may carry an error of its own. It is one plain type whose parameters are fields, so a graph
and everything that gives its codes meaning serialize together; nothing in it is a closure.

An estimator implements:

- `encode(est, obj)`: what is stored for the raw `obj`, the code plus whatever the
  estimator keeps with it (a norm, a correction term, a finer code);
- `encodequery(est, q)`, when the query side needs preparing: what `evaluate` takes as its
  query for the raw `q`. A rotation, for one -- applying it inside `evaluate` would cost
  `D^2` per pair against the `D` of the estimate, so the graph applies it once per query
  and once per inserted item. The default is the identity;
- `evaluate(est, q, stored)`: the distance between the raw query `q` and a stored object,
  in that order -- every index here evaluates its query first. It receives everything the model has -- the raw query, the encoded object with what was
  kept beside the code, and the estimator's own parameters -- so a model that can bound its
  error re-evaluates *inside* the evaluation when it must, and returns the distance it
  stands behind. The graph only ever evaluates the distance; whether that was an estimate,
  a corrected estimate or a re-evaluation is the estimator's business.
- `evaluate(est, a, b)` between two **stored** objects as well: the neighborhood filters
  (`SatNeighborhood` and its relatives) compare a new item's candidates among themselves to
  decide which edges to keep, and those candidates are stored objects. That is the
  symmetric estimate, code against code, and it only shapes the edges; the scalar
  quantizers' distances already have it.

The no-op estimator is a plain distance: `ScalarQuant.SqL2()` against an `SQVec` evaluates
the query against the codes and nothing needs correcting. Any `PreMetric` works as the
distance of an asymmetric graph; this type is the documented home for the ones that encode.
"""
abstract type AbstractEstimator <: PreMetric end

"""
    rotate(rot, v) -> Vector{Float32}

Applies the rotation `rot` to `v`, returning a fresh `Vector{Float32}` of the same norm. It is
the one thing an estimator that lives in a rotated space asks of its rotation, and the graph
pays it once per query and once per inserted item (`encodequery`, `encode`), never per
evaluation. `nothing` is the rotation that rotates nothing and only casts; a square
`Projections.RandomProjections` (`Projections.qr(dim, dim)`) and a
`Projections.RandomizedHadamard` implement it (see [`Projections.Rotation`](@ref)), and any
other type may, together with [`rotationdim`](@ref).
"""
function rotate end
rotate(::Nothing, v::AbstractVector) = Vector{Float32}(v)

"""
    rotationdim(rot) -> Int

The dimension `rot` rotates, so an estimator reads it off its rotation and checks its data
against it. A `RandomProjections` that is not square throws: a projection that changes the
dimension is not a rotation.
"""
function rotationdim end

"How `rot` is named in an estimator's `show`: the type's name, or `nothing`."
rotationname(::Nothing) = "nothing"
rotationname(rot) = String(nameof(typeof(rot)))
