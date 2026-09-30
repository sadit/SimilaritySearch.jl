```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

## Indexes

```@docs
ExhaustiveSearch
ParallelExhaustiveSearch
AbstractSearchGraph
SearchGraph
AsymmetricSearchGraph
AbstractEstimator
SimilaritySearch.encode
SimilaritySearch.encodequery
SimilaritySearch.rotate
SimilaritySearch.rotationdim
BKT
PermutedSearchIndex
distance
```

## Searching

```@docs
search
searchbatch
searchbatch!
```

## Computing all knns
The operation of computing all knns in the index is computed as follows:
```@docs
allknn
```

## Computing closest pair(s), and the bichromatic metric join (`Bichromatic` submodule)
The operation of finding the closest pair of elements in the indexed dataset, its bichromatic
counterpart (the closest pair between an indexed dataset and another dataset), their `k`-pairs
generalizations, and a metric join between two datasets when neither the match count per element nor
a join radius is known ahead of time.
```@docs
closestpair
bichromatic_closestpair
closestpairs
bichromatic_kclosestpairs
bichromatic_metricjoin
```

## Selection: picking a subset that stands for the whole dataset
Two dual shapes. The fixed-count selectors are told how many centers to pick and the radius
they achieve falls out; `neardup` is told the radius and the count falls out. All of them
report `centers`/`assign`/`assigndist` under the same names -- see [`AbstractSelection`](@ref).
```@docs
Selection
AbstractSelection
fft
dnet
randsel
multirandsel
CenterSelection
neardup
NearDupSelection
```

## Other high level algorithms
```@docs
hsp_queries
rerank!
distsample
distsample_ut
recallscore
macrorecall
```

## Parallel batching (`@BATCHES`)
The primitive every batch operation above (`searchbatch`, `allknn`, `closestpair`,
`neardup`, `index!`, the k-centers algorithms, ...) is built on; see the
[parallelism tutorial](@ref "Parallelism: what to expect, what not to do") for a guided
introduction, including the `:sequential` scheduler and how contexts carry their own
`scheduler`.
```@docs
@BATCHES
@BEGIN
@BEGINBATCH
@LOOP
@ENDBATCH
@END
@batchid
@nbatches
set_batch_scheduler!
get_batch_scheduler
```

## Indexing elements
```@docs
push_item!
append_items!
index!
rebuild
```

## Logging
A context carries two logging slots: `ctx.reporters`, where progress messages go to be
read, and `ctx.observers`, what reacts to a structural change so that something durable
happens. `reporters=[]` silences a context completely without disturbing observation. See
the [logging tutorial](@ref "Reporting, observing, and capturing neighbors as they're built")
for worked examples of both.
```@docs
AbstractLog
AbstractReporter
AbstractObserver
INFORM
@inform
InformativeLog
OBSERVE
CallbackLog
LOG
```

## Distance functions
The distance functions are defined to work under the `evaluate(::metric, u, v)` function (borrowed from [Distances.jl](https://github.com/JuliaStats/Distances.jl) package). None of them are re-exported from `SimilaritySearch` directly; access them through the `Dist` submodule, e.g. `Dist.L2()`.

### Minkowski vector distance functions
```@docs
Dist.L1
Dist.L2
Dist.SqL2
Dist.LInfty
Dist.Lp
```

### Cosine and angle distance functions for vectors
```@docs
Dist.Cosine
Dist.NormCosine
Dist.Angle
Dist.NormAngle
```

### Set distance functions
Set objects are represented as ordered arrays, accessed via `Dist.Sets`.
```@docs
Dist.Sets.Jaccard
Dist.Sets.Dice
Dist.Sets.Intersection
Dist.Sets.CosineSet
Dist.Sets.RogersTanimoto
```

### Bit-vector distance functions
Accessed via `Dist.Bits`.
```@docs
Dist.Bits.Hamming
Dist.Bits.RogersTanimoto
Dist.Bits.RussellRao
```

### String and sequence alignment distances
The following uses strings/arrays as input, i.e., objects follow the array interface. Accessed via `Dist.Seqs`. A broader set of distances for strings can be found in the [StringDistances.jl](https://github.com/matthieugomez/StringDistances.jl) package.

```@docs
Dist.Seqs.CommonPrefix
Dist.Seqs.Levenshtein
Dist.Seqs.DamerauLevenshtein
Dist.Seqs.Hamming
Dist.Seqs.LCS
```

### Distances for clouds of points
Accessed via `Dist.Cloud`.
```@docs
Dist.Cloud.Hausdorff
Dist.Cloud.DirectedHausdorff
Dist.Cloud.Chamfer
Dist.Cloud.EMD
```

### Distance wrappers and hacks
Accessed via `Dist.Hacks`.
```@docs
Dist.Hacks.NegativeDistanceHack
Dist.Hacks.SimilarityFromDistance
Dist.Hacks.DistanceWithIdentifiers
```

## Functions that customize parameters
Several algorithms support arguments that modify the performance, for instance, some of them should be computed or prepared with external functions or structs

```@docs
getminbatch
AbstractContext
GenericContext
SearchGraphContext
BeamSearch
BeamSearchSpace
OptimizeParameters
optimize_index!
MinRecall
OptRadius
ParetoRecall
ParetoRadius
```

### Neighborhood computation and refinement
```@docs
Neighborhood
NeighborhoodFilter
IdentityNeighborhood
SatNeighborhood
DistalSatNeighborhood
KCentersNeighborhood
find_neighborhood!
```

### Hints (entry points for approximate search)
```@docs
RandomHints
DisjointHints
KDisjointHints
EpsilonHints
KCentersHints
AdjacentStoredHints
matrixhints
```

### Callbacks
```@docs
Callback
execute_callbacks!
```

## Database API
```@docs
AbstractDatabase
MatrixDatabase
BlockMatrixDatabase
MMapMatrixDatabase
VectorDatabase
SubDatabase
```

## Adjacency list API
The backing storage for a [`SearchGraph`](@ref)'s edges.
```@docs
AbstractAdjList
AdjList
AdjDict
StaticAdjList
```

## k-NN and radius-bounded result containers (`PQueue` submodule)
Result containers accumulate `(id, dist)` pairs found during a search. They live under
`AbstractMetricQueue`, with two sibling families: count-bounded (`AbstractKnnQueue`: `KnnHeap`,
`KnnSorted`, keep the `k` closest items) and radius-bounded (`AbstractRadiusQueue`:
`RadiusSorted`, `RadiusHeap`, keep every item within a fixed distance threshold, however many
that turns out to be -- see the [`searchbatch!`](@ref) form that accepts a vector of these).
A radius container is a result container only: graph searches over one are navigated with the
internal `SimilaritySearch.BallKnn`, which adds the navigation reserve they lack, and receive just
its in-ball part.
Although they're implemented in the `PQueue` submodule, every name below is re-exported
unqualified from `SimilaritySearch`, exactly as before this reorganization.
```@docs
AbstractMetricQueue
AbstractKnnQueue
AbstractRadiusQueue
KnnHeap
KnnSorted
RadiusSorted
RadiusHeap
SimilaritySearch.BallKnn
SimilaritySearch.ballview
knnqueue
nearest
frontier
covradius
reuse!
sortitems!
sort_last_item!
maxlength
isheap
heapsort!
heapfix_down!
pop_min!
pop_max!
IdDist
IdOrder
DistOrder
RevDistOrder
IdView
DistView
IdDistView
knn_matrices
```

## Scalar quantization (`ScalarQuant` submodule)
Reduces the memory footprint of a database by quantizing each coordinate to 2, 4 or 8
bits, in two families that differ in where the quantization range comes from (see the
module docstring for how to choose). Both produce the same vector type, `SQVec`, and share
one set of distances that read the packed codes directly; the per-width submodules keep
their un-prefixed API (`ScalarQuant.SQu8.quantize`, `ScalarQuant.SQu8.SqL2`, ...) as aliases.
```@docs
ScalarQuant
ScalarQuant.SQMinC
ScalarQuant.SQVec
ScalarQuant.quantvector!
```

### The quantized database (`QuantDatabase`)
One database type for both families, over any storage of `UInt8` vectors: static
(`MatrixDatabase`), growing in blocks (`BlockMatrixDatabase`), on disk (`MMapMatrixDatabase`)
or anything indexable. `push_item!`/`append_items!` quantize on the way in with the
database's own parameters.
```@docs
ScalarQuant.QuantDatabase
ScalarQuant.quantize
ScalarQuant.isglobal
ScalarQuant.codewidth
```

### Distances over quantized vectors
Defined once for every width and both families. Between two quantized vectors each is one
integer pass over the codes plus the per-vector sums; against a plain `Float32` vector the
codes are unpacked to floats with SIMD.
```@docs
ScalarQuant.SqL2
ScalarQuant.L2
ScalarQuant.L1
ScalarQuant.NormCosine
ScalarQuant.Cosine
```

### Per-column quantization (`SQu2`, `SQu4`, `SQu8` submodules)

Each column (vector) keeps its own `min`/scale, computed from its own extrema.
```@docs
ScalarQuant.SQu2
ScalarQuant.SQu2.quantize
ScalarQuant.SQu2.SQu2Database
ScalarQuant.SQu4
ScalarQuant.SQu4.quantize
ScalarQuant.SQu4.SQu4Database
ScalarQuant.SQu8
ScalarQuant.SQu8.quantize
ScalarQuant.SQu8.SQu8Database
```

### Global (database-wide) quantization (`SQgu2`, `SQgu4`, `SQgu8` submodules)

All columns share a single `min`/scale, letting the distance kernels compare the packed
codes directly with SIMD, without any per-element dequantization. Each submodule offers an
allocating `quantize` and an in-place `quantize!(vout, v, minmax)` for loops that reuse
their output buffer.
```@docs
ScalarQuant.sqglobalscale
ScalarQuant.sqautorange
ScalarQuant.sqrange
ScalarQuant.SQgu2
ScalarQuant.SQgu2.quantize
ScalarQuant.SQgu2.quantize!
ScalarQuant.SQgu2.SqL2
ScalarQuant.SQgu4
ScalarQuant.SQgu4.quantize
ScalarQuant.SQgu4.quantize!
ScalarQuant.SQgu4.SqL2
ScalarQuant.SQgu8
ScalarQuant.SQgu8.quantize
ScalarQuant.SQgu8.quantize!
ScalarQuant.SQgu8.SqL2
```

### A database that keeps its quantization parameters (`GlobalQuantDatabase`)

`SQgu*.quantize` returns a bare matrix of codes and leaves `min`/`max` to the caller, so
stored codes cannot be dequantized and can only be compared against codes from the same run.
`GlobalQuantDatabase` keeps the pair, and the per-vector code sums an order-preserving cosine
needs; it is the global-family `QuantDatabase`, so every distance above applies to it and it
grows like any other.
```@docs
ScalarQuant.GlobalQuantDatabase
```

### The quantizers as an encoder for the asymmetric graph (`SQEncoder`)

Objects are quantized once and stored as codes, queries are kept in `Float32`, and the
distances above evaluate one against the other; an optional rotation is applied to both
sides first. It uses the `AbstractEstimator` interface the `AsymmetricSearchGraph` navigates
with, but carries no error model. The quantizer is named by its module (`SQgu4`, `SQu8`, ...),
the rotation by the object that applies it (`Projections.qr(dim, dim)`,
`Projections.RandomizedHadamard(dim)`) or `nothing`.
```@docs
ScalarQuant.SQEncoder
ScalarQuant.sqcodes
ScalarQuant.quantizer
```

## RaBitQ (`RaBitQ` submodule)

The RaBitQ estimator (Gao & Long, 2024) as an `AbstractEstimator`: sign bits of the rotated
vector plus three scalars per object, an unbiased estimate of the cosine with a per-object
error bound, and a two-level variant that keeps a fallback beside the bits and re-evaluates
from it, inside the estimate, when the bound cannot rule an object out.
```@docs
RaBitQ
RaBitQ.AbstractRaBitQ
RaBitQ.RaBitQCode
RaBitQ.RaBitQQuery
RaBitQ.rabitqcodes
RaBitQ.estimatecos
RaBitQ.errorbound
RaBitQ.RaBitQRefined
RaBitQ.AbstractFallback
RaBitQ.RaBitQExactFallback
RaBitQ.RaBitQVectorFallback
RaBitQ.refinethreshold
```

## Random projections (`Projections` submodule)
```@docs
Projections.RandomProjections
Projections.gaussian
Projections.qr
Projections.outdim
Projections.indim
Projections.transform
Projections.transform!
Projections.bitsketch
```

## Hadamard projection (`Projections.HadamardProjection`) and the rotations

A projection computed with the fast Walsh-Hadamard transform
(via [Hadamard.jl](https://github.com/stevengj/Hadamard.jl)'s `fwht_natural!`) instead of a dense
random matrix. Uses the same `outdim`/`indim`/`transform`/`transform!`/`bitsketch` generic
functions documented above for `RandomProjections`. `RandomizedHadamard` makes a random
rotation of it (a random sign per coordinate first, and the norm preserved), and `Rotation`
is what the estimators (`ScalarQuant.SQEncoder`, `RaBitQ`) take as theirs.

```@docs
Projections.HadamardProjection
Projections.RandomizedHadamard
Projections.Rotation
```

## PCA projection (`Projections.PCAProjection`)

A projection fitted from data, via [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl)'s
`PCA`, instead of a random or structured rotation. Uses the same
`outdim`/`indim`/`transform`/`transform!`/`bitsketch` generic functions documented above
for `RandomProjections`; unlike those, its matrix `transform` has no `minbatch` (a single
vectorized call into MultivariateStats already covers every column).

```@docs
Projections.PCAProjection
```

## Hyperplane bit sketches (`Projections` submodule)

Binary sketch generators for *any* metric space -- not just floating-point vectors under
`transform` above: an object is encoded by which side of a set of hyperplanes, pairs of
anchor objects compared through the space's own distance function, it falls on. Each of
these carries its own [`distance`](@ref) (Hamming, over the packed sketch) and supports
[`Projections.outdim`](@ref)/[`Projections.bitsketch`](@ref) like the projections above.
See the [bit sketches tutorial](@ref "Quantization and Bit Sketches") for a worked example.

```@docs
Projections.DistantHyperplanes
Projections.AnchoredDistantHyperplanes
Projections.RandomHyperplanes
```

## Multi-bit sketches (`Projections.QuantSketch`)

The same sketch models as above, but keeping an `m`-bit unsigned code per component
(`m = 2, 4, 8`, via [`ScalarQuant.SQgu2`](@ref)/[`ScalarQuant.SQgu4`](@ref)/[`ScalarQuant.SQgu8`](@ref))
instead of a single sign bit -- so a sketch records *how far* an object sits from each
hyperplane, not merely on which side. `nbits=1` is supported too and reproduces
[`Projections.bitsketch`](@ref) exactly, so a sweep over `1, 2, 4, 8` bits runs through one
API. Applies to both families: for a rotation the encoded value is the projected
coordinate, for a metric hyperplane it is the signed margin
`d(obj, b) - d(obj, a)` -- see [`Projections.sketchvalues!`](@ref).

```@docs
Projections.QuantSketch
Projections.quantsketch
Projections.sketchvalues!
Projections.sketchbits
Projections.sketchsize
Projections.hyperplanewidths
```

## Sketch-based search pipeline (`Projections.SketchedSearch`)

Encode, index, retrieve candidates cheaply, re-score them exactly -- packaged as an
ordinary `AbstractSearchIndex`, so `search`/`searchbatch` work on it unchanged and its
results are ids into the original database with true distances.

```@docs
Projections.SketchedSearch
Projections.exhaustivesketchindex
```

## Spherical embedding for MIPS (`Special.Spherical` submodule)

Turns Maximum Inner Product Search into ordinary nearest-neighbor search (Neyshabur &
Srebro's asymmetric spherical embedding), for dense and sparse vectors alike.

```@docs
Special.Spherical
Special.Spherical.SphericalEmbedding
Special.Spherical.outdim
Special.Spherical.indim
Special.Spherical.transform
Special.Spherical.transform!
Special.Spherical.transform_query
Special.Spherical.transform_query!
```

## Sparse vector support (`Special.Sparse` submodule)

A sparse matrix view tailored for distance evaluations, replacing Base's `SparseVector`
with an explicit dimension-tracking read-only wrapper `SparseVecView`.

```@docs
Special.Sparse
Special.Sparse.SparseVecView
Special.Sparse.SparseDatabase
Special.Sparse.sparsedot
```


## Inverted files (`InvertedFiles` submodule)

Inverted file index data structures and context for sparse vectors, MIPS, and set search.

```@docs
InvertedFiles.AbstractInvertedFile
InvertedFiles.InvertedFile
InvertedFiles.DictInvertedFile
InvertedFiles.InvertedFileContext
InvertedFiles.getcontext
InvertedFiles.search_invfile
InvertedFiles.select_posting_lists
InvertedFiles.SortedIntSet
```

## Posting list intersections (`Intersections` submodule)

Algorithms for set and posting list intersections.

```@docs
Intersections.svs
Intersections.bk!
Intersections.bkt!
Intersections.umerge!
Intersections.imerge!
Intersections.xmerge!
```

