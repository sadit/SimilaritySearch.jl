```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```


# Distance functions

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

### Distances that cast to `Float32` (`Dist.CastF32` submodule)

The same distances, with the pair converted to `Float32` before the kernel runs. Use them when
the database stores a narrower type, such as `Float16`, and the arithmetic should still happen in
single precision.

```@docs
Dist.CastF32.L1
Dist.CastF32.L2
Dist.CastF32.SqL2
Dist.CastF32.Lp
Dist.CastF32.LInfty
Dist.CastF32.Cosine
Dist.CastF32.NormCosine
Dist.CastF32.Angle
Dist.CastF32.NormAngle
```
