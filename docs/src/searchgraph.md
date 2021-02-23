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


index = SearchGraph(L2Distance(), X, verbose=true)

for q in Q
    println(search(index, q, KnnResult(3)))
end
```

There are a several options for the construction of the index, from specifying the precise searching algorithm (`search_algo`) and the neighborhood strategy to be used on construction (`neighborhood_algo`). You can also specify here if the construction will be made in parallel and how this paralllelism will be performed (`firstblock`, `block`).

```@docs

SearchGraph
SearchGraphOptions

```

Regarding the search search algorithm, there are two alternatives, based on local search heuristics.

```@docs

BeamSearch
IHCSearch

```

Regarding neighborhood algorithms, the package defines the following ones:

Neighborhood algorithms
```@docs

FixedNeighborhood
LogNeighborhood
LogSatNeighborhood
SatNeighborhood

```

----
The index can be created incrementally,

```@example
using SimilaritySearch

index = SearchGraph(L2Distance(), Vector{Float32}[])

for i in 1:1000
    push!(index, rand(Float32, 4))
end

search(index, rand(Float32, 4), 3)
```

The index construction can be customized with `find_neighborhood` and `push_neighborhood`
```@docs
push!(::SearchGraph, elem)
find_neighborhood
push_neighborhood!

```

```@docs
optimize!
```

also see search and push!