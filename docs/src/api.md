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
SearchGraph
PermutedSearchIndex
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

## Remove near duplicates
Finds and removes near duplicate items in a metric dataset
```@docs
neardup
```

## Other high level algorithms
```@docs
hsp_queries
rerank!
fft
dnet
randsel
multirandsel
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
IdIntDist
IdOrder
DistOrder
RevDistOrder
IdView
DistView
IdDistView
knn_matrices
```

## Scalar quantization (`ScalarQuant` submodule)
Reduces the memory footprint of a database by quantizing each coordinate to a small
integer type. Each bit-width/strategy lives in its own nested submodule with a common,
un-prefixed API (`quantize`, `L1`, `L2`, `SqL2`, `NormCosine`), accessed e.g. as
`ScalarQuant.SQu8.quantize`, `ScalarQuant.SQu8.SqL2`, etc.

### Per-column quantization (`SQu2`, `SQu4`, `SQu8` submodules)

Each column (vector) keeps its own `min`/scale, computed from its own extrema.
```@docs
ScalarQuant.SQu2
ScalarQuant.SQu2.quantize
ScalarQuant.SQu2.SQu2Vec
ScalarQuant.SQu2.SQu2Database
ScalarQuant.SQu2.L1
ScalarQuant.SQu2.L2
ScalarQuant.SQu2.SqL2
ScalarQuant.SQu4
ScalarQuant.SQu4.quantize
ScalarQuant.SQu4.SQu4Vec
ScalarQuant.SQu4.SQu4Database
ScalarQuant.SQu4.L1
ScalarQuant.SQu4.L2
ScalarQuant.SQu4.SqL2
ScalarQuant.SQu8
ScalarQuant.SQu8.quantize
ScalarQuant.SQu8.SQu8Vec
ScalarQuant.SQu8.SQu8Database
ScalarQuant.SQu8.L1
ScalarQuant.SQu8.L2
ScalarQuant.SQu8.SqL2
ScalarQuant.SQu8.NormCosine
```

### Global (database-wide) quantization (`SQgu4`, `SQgu8` submodules)

All columns share a single `min`/scale, letting the distance kernels compare the packed
codes directly with SIMD, without any per-element dequantization.
```@docs
ScalarQuant.SQgu4
ScalarQuant.SQgu4.quantize
ScalarQuant.SQgu4.NormCosine
ScalarQuant.SQgu4.SqL2
ScalarQuant.SQgu8
ScalarQuant.SQgu8.quantize
ScalarQuant.SQgu8.NormCosine
ScalarQuant.SQgu8.SqL2
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
```

## Hadamard projection (`Projections.HadamardProjection`)

A projection computed with the fast Walsh-Hadamard transform
(via [Hadamard.jl](https://github.com/stevengj/Hadamard.jl)'s `fwht_natural!`) instead of a dense
random matrix. Uses the same `outdim`/`indim`/`transform`/`transform!` generic functions
documented above for `RandomProjections`.

```@docs
Projections.HadamardProjection
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

