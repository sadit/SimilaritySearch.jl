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

## Computing closest pair
The operation of finding the closest pair of elements in the indexed dataset.
```@docs
closestpair
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
distsample
distsample_ut
recallscore
macrorecall
```

## Indexing elements
```@docs
push_item!
append_items!
index!
rebuild
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
Dist.Cloud.Chamfer
Dist.Cloud.EMD
```

### Distance wrappers and hacks
Accessed via `Dist.Hacks`.
```@docs
Dist.Hacks.NegativeDistanceHack
Dist.Hacks.SimilarityFromDistance
Dist.Hacks.DistanceWithIdentifiers
Dist.Hacks.DistanceF32
```

## Functions that customize parameters
Several algorithms support arguments that modify the performance, for instance, some of them should be computed or prepared with external functions or structs

```@docs
getminbatch
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
StrideMatrixDatabase
BlockMatrixDatabase
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

## k-NN result containers
```@docs
AbstractKnn
KnnHeap
KnnSorted
knnqueue
nearest
frontier
viewitems
covradius
reuse!
sortitems!
maxlength
IdDist
IdIntDist
IdOrder
DistOrder
RevDistOrder
```

## Scalar quantization (`ScalarQuant` submodule)
Reduces the memory footprint of a database by quantizing each coordinate to a small integer type. Accessed as `ScalarQuant.SQu2`, etc.
```@docs
ScalarQuant.SQu2
ScalarQuant.SQu2Vec
ScalarQuant.SQu4
ScalarQuant.SQu4Vec
ScalarQuant.SQu8
ScalarQuant.SQu8Vec
ScalarQuant.sq_global_u8
```

## Random projections (`Special.Projections` submodule)
```@docs
Special.Projections.RandomProjections
Special.Projections.gaussian
Special.Projections.qr
Special.Projections.outdim
Special.Projections.indim
Special.Projections.transform
Special.Projections.transform!
```
