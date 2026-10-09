```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```


# API, core

The index types, the search entry points, and everything a query needs around them. The rest of
the API is split by area: [distances](api_distances.md),
[quantization and sketches](api_quantization.md),
[inverted files and sparse data](api_invertedfiles.md), and [tree indexes](api_trees.md).

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
PermutedSearchIndex
distance
AbstractSearchIndex
InsertionSource
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
allknn!
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
```

## Scores: recall, match error, and their bootstrap

A score is a two-argument function of a query's gold result and the result to evaluate; the
macro score is its mean over the queries, and [`bootstrapscore`](@ref) resamples the queries
to say how much that mean would move. [`MinRecall`](@ref) and [`MaxMatchError`](@ref) are
built on `macrorecall` and `macromatcherror`.
```@docs
recallscore
macrorecall
matcherror
macromatcherror
perqueryscores
bootstrapscore
BootstrapScore
```

## Parallel batching (`@BATCHES`)
The primitive every batch operation above (`searchbatch`, `allknn`, `closestpair`,
`neardup`, `index!`, the k-centers algorithms, ...) is built on; see the
[parallelism tutorial](@ref "Parallelism and Multithreading") for a guided
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
beginbatch
distance_evaluations
distance_stats
block_evaluations
block_stats
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
the [logging tutorial](@ref "Logging and Observation Channels")
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
verbose
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
MaxMatchError
SimilaritySearch.goalvalue
ErrorFunction
LocalSearchAlgorithm
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

### Near duplicates: members, `expand` and `expand!`

A graph built with `Neighborhood(neardup=ϵ)` keeps one node per cluster of near duplicates
and makes the rest *members* of it: never visited, never answered by `search`, which returns
representatives. `expand` and `expand!` are the second stage, from `k` clusters to the raw
neighbors, on any result form. A non-negative `ϵ` -- here and in [`neardup`](@ref) -- is raised
to [`NEARDUP_NUMERICAL_ZERO`](@ref SimilaritySearch.NEARDUP_NUMERICAL_ZERO), since two identical objects usually do not evaluate to exactly
`0f0`; a negative one is rejected.
```@docs
Members
members
representative
ismember
expand
expand!
SimilaritySearch.NEARDUP_NUMERICAL_ZERO
addmember!
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
SimilaritySearch.prefetch_item
SimilaritySearch.prefetchable
VectorDatabase
SubDatabase
database
```

## Adjacency list API
The backing storage for a [`SearchGraph`](@ref)'s edges.
```@docs
AbstractAdjList
AdjList
AdjDict
StaticAdjList
neighbors
neighbors_length
add!
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
PQueue.heapify!
PQueue.ninside
PQueue.heapfix_up!
```

```@docs
SimilaritySearch.ScalarQuant.RangePolicy
SimilaritySearch.ScalarQuant.SymmetricRange
SimilaritySearch.ScalarQuant.ExtremaRange
```
