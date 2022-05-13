```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

## Indexes

```@docs
ExhaustiveSearch
SearchGraph
```

## Searching

```@docs

search
searchbatch
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

## Indexing elements
```@docs
push!(::SearchGraph, item)
push_item!
append!
index!
```

## Distance functions
The distance functions are defined to work under the `evaluate(::metric, u, v)` function (borrowed from [Distances.jl](https://github.com/JuliaStats/Distances.jl) package).

### Cosine and angle distances for vectors
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

## Public API

```@autodocs
Modules = [SimilaritySearch]
Private = false
Order = [:function, :type]
```

## Private API

```@autodocs
Modules = [SimilaritySearch]
Public = false
Order = [:function, :type]
```