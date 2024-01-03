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
```

## Searching

```@docs
search
searchbatch
```

Note: `KnnResult` based functions are significantly faster in general on pre-allocated objects that similar functions accepting matrices of identifiers and distances. Matrix based outputs are based on `KnnResult` methods that copy their results on the matrices.
Preallocation is also costly, so if you have relatively small datasets, you are not intended to repeat the search process many times, or you are unsure, it is safe to use matrix-based functions.

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

## Indexing elements
```@docs
push_item!
append_items!
index!
rebuild
```


## Distance functions
The distance functions are defined to work under the `evaluate(::metric, u, v)` function (borrowed from [Distances.jl](https://github.com/JuliaStats/Distances.jl) package).

### Minkowski vector distance functions
```@docs
L1Distance
L2Distance
SqL2Distance
LInftyDistance
LpDistance
```

The package implements some of these functions using the `@turbo` macro from [`LoopVectorization`](https://github.com/JuliaSIMD/LoopVectorization.jl) package.
```@docs
TurboL1Distance
TurboL2Distance
TurboSqL2Distance
TurboNormalizedCosineDistance

```

### Cosine and angle distance functions for vectors
```@docs
CosineDistance
NormalizedCosineDistance
AngleDistance
NormalizedAngleDistance
```

### Set distance functions
Set bbject are represented as ordered arrays
```@docs
JaccardDistance
DiceDistance
IntersectionDissimilarity
CosineDistanceSet
```

### String alignment distances
The following uses strings/arrays as input, i.e., objects follow the array interface. A broader set of distances for strings can be found in the [StringDistances.jl](https://github.com/matthieugomez/StringDistances.jl) package.

```@docs
CommonPrefixDissimilarity
GenericLevenshteinDistance
StringHammingDistance
LevenshteinDistance
LcsDistance
```

### Distances for Cloud of points

```@docs
HausdorffDistance
MinHausdorffDistance
```

## Functions that customize parameters
Several algorithms support arguments that modify the performance, for instance, some of them should be computed or prepared with external functions or structs

```@docs
getminbatch
getknnresult
getpools
Neighborhood
Callback
SearchGraphCallbacks
BeamSearchSpace
```

## Database API
```@docs
AbstractDatabase
MatrixDatabase
VectorDatabase
DynamicMatrixDatabase
StrideMatrixDatabase
```

```@docs

find_neighborhood
push_neighborhood
SatPruning
RandomPruning
KeepNearestPruning
NeighborhoodPruning
maxlength
get_parallel_block
SimilarityFromDistance
execute_callbacks

```
