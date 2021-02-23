```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

This package focus on solving ``k``-NN queries, that is, retrieving the ``k`` nearest neighbors of a given query in a collection of items under a distance function. The distance function is often a metric function.

The general procedure is as follows:

```@example example1
using SimilaritySearch
X = [randn(3) for i in 1:10_000]
Q = [randn(3) for i in 1:5]


index = ExhaustiveSearch(L2Distance(), X)

[search(index, q, KnnResult(3)) for q in Q]

```

Given a dataset ``X``, we need to create an index structure; in this example, we created a sequential search index named [`ExhaustiveSearch`](@ref). The search is performed by [`search`](@ref). This function recieves the index, the query and the specification of the query (i.e., the [`KnnResult`](@ref) object, which also works as container of the result set).

Regarding the distance function, [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl) defines several distance functions, but also can work with any of the distance functions specified in [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) package.

## Other similarity search indexes
The [`ExahustiveSearch`](@refs) index performs an exhaustive evaluation of the query against each element of the dataset, but without any preprocessing cost.

In addition of this, the package implements other indexes that can improve the search cost in diverse situations. These indexes have memory and preprocessing time requirements that must be considered in any real application.

### Approximate search
Performs approximate search; they could solve the query, i.e., the result can lost some items or include some others not being part of the exact solution. In contrast, these indexes are often quite faster and more flexible than exact search methods.

- [`SearchGraph`](@ref): Very fast and precise similarity search index; supports multithreading construction (**cite**).
- [`Knr`](https://github.com/sadit/NeighborhoodApproximationIndex.jl): Indexes based on K nearest references (_external package_, **cite**).
- [`DeloneInvIndex`](https://github.com/sadit/NeighborhoodApproximationIndex.jl): An index based on a delone partition (_external package_, **cite**).

### Exact search
- [`Kvp`](@ref): K vantage points (**cite**).
- [`PivotedSearch`](@ref): A generic pivot table (**cite**).
- [`sss`](@ref): A pivot table with pivots selected with the SSS scheme (**cite**).
- [`distant_tournament`](@ref). A pivot table where pivots are selected using a simple distant tournament (**cite**).

