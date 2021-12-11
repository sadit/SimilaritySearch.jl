```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

`SearchGraph` creates a graph

_Tellez, E. S., Ruiz, G., Chavez, E., & Graff, M.A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs. Pattern Analysis and Applications, 1-15._


As before, we need a dataset ``X`` and a distance function.

```@example
using SimilaritySearch
X = [randn(3) for i in 1:10_000]
Q = [randn(3) for i in 1:5]


index = SearchGraph(; dist=L2Distance(), verbose=true)
append!(index, X)

for q in Q
    println(search(index, q, KnnResult(3)))
end
```

There are a several options for the construction of the index, from specifying the precise searching algorithm (`search_algo`) and the neighborhood strategy to be used on construction (`neighborhood_algo`). You can also specify here if the construction will be made in parallel and how this paralllelism will be performed (`firstblock`, `block`).

```@docs

SearchGraph
SearchGraphOptions

```

## Local search algorithms
Regarding the search search algorithm, there are two alternatives, based on local search heuristics.

```@docs

BeamSearch


```

## Neighborhood algorithms
Regarding neighborhood algorithms, the package defines the following ones:

Neighborhood algorithms
```@docs

FixedNeighborhood
LogNeighborhood
LogSatNeighborhood
SatNeighborhood

```

## Incremental construction of the index
The index supports insertions; for instance, it can be fully constructed using insertions as follows:

```@example
using SimilaritySearch

index = SearchGraph()

append!(index, [rand(Float32, 4) for i in 1:1000])

search(index, rand(Float32, 4), 3)
```

Each insertion is performed by the `push!` function or a chunk of items with `append!`; the last function also allows parallel insertions.
```@docs
push!(::SearchGraph, elem)
append!(::SearchGraph, elem)
```

### Customizing the insertion method
In particular, the procedure of pushing items into the index is made calling `find_neighborhood` and `push_neighborhood`; these methods can be overriden or used in other wyas to customize the insertion of data and the construction of the index.

```@docs
find_neighborhood
push_neighborhood!

```
### Optimizing the index's performance
```@docs
optimize!
```

