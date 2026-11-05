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

## Singleton indexes (`Anni` submodule)

Exact indexes for metrics over a Dirichlet partition whose number of centers tunes itself on a
sample of queries (Tellez, Ruiz and Chávez, Information Systems 60, 2016).

```@docs
Anni.ANNI
Anni.NANNI
Anni.getcontext
Anni.annipermutation
Anni.permuteanni
Anni.nannipermutation
Anni.permutenanni
```

## Extreme pivots (`ExtremePivots` submodule)

An exact pivot table that keeps, per object and group, only the pivot that sees it farthest
from its mean distance (Ruiz, Chávez, Ruiz and Tellez, Knowledge and Information Systems, 2020),
and its node-based version, where the nearest pivot defines the cell and the far ones filter
inside it.

```@docs
ExtremePivots.EPTable
ExtremePivots.NEP
ExtremePivots.getcontext
ExtremePivots.eptpermutation
ExtremePivots.permuteept
ExtremePivots.neppermutation
ExtremePivots.permutenep
```

## Stretched triangle inequality

The exact indexes above made approximate by a factor on the covering radius of every filter,
tuned like the `Δ` of a beam search (Chávez and Navarro, Information Processing Letters 85, 2003).

```@docs
PruningSearch
PruningSearchSpace
```
