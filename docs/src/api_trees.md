```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```


# Tree-based indexes

Two indexes build a tree instead of a graph. Both are exact and both insert incrementally.

## Spatial access tree (`SpatialAccessTree` submodule)

```@docs
SpatialAccessTree.Sat
SpatialAccessTree.SatContext
SpatialAccessTree.getcontext
SpatialAccessTree.BeamSearchSat
SpatialAccessTree.BeamSearchParSat
SpatialAccessTree.BeamSearchMultiSat
SpatialAccessTree.PruningSat
SpatialAccessTree.PrunParSat
SpatialAccessTree.PruningSatSpace
SpatialAccessTree.permutesat
SpatialAccessTree.satpermutation
SpatialAccessTree.satpermutation!
SpatialAccessTree.SatInitialPartition
SpatialAccessTree.RandomInitialPartition
SpatialAccessTree.ProximalSortSat
SpatialAccessTree.DistalSortSat
SpatialAccessTree.RandomSortSat
```

## BK-tree (`BKTree` submodule)

An exact index for discrete metrics, where a graph has no usable distance gradient.

```@docs
BKTree.BKT
BKTree.getcontext
```
