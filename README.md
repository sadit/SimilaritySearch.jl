[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/SimilaritySearch.jl/dev)
[![Build Status](https://github.com/sadit/SimilaritySearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/SimilaritySearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/SimilaritySearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/SimilaritySearch.jl)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04442/status.svg)](https://doi.org/10.21105/joss.04442)

# SimilaritySearch.jl

SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for `SearchGraph,` a fast and flexible search index using any metric function. It is designed to support multithreading in most of its functions and structures.

The package provides the following indexes:

- `ParallelExhaustiveSearch`: A brute force search index where each query is solved using all available threads.
- `ExhaustiveSearch`: A brute force search index, each query is solved using a single thread.
- `SearchGraph`: An approximate search index with parallel construction.

The main set of functions are:

- `search`: Solves a single query.
- `searchbatch`: Solves a set of queries.
- `allknn`: Computes the $k$ nearest neighbors for all elements in an index.
- `neardup`: Removes near-duplicates from a metric dataset.
- `closestpair`: Computes the closest pair in a metric dataset.

The precise definitions of these functions and the complete set of functions and structures can be found in the [documentation](https://sadit.github.io/SimilaritySearch.jl/dev).

# Similarity search _ecosystem_ in Julia
Currently, there exists several packages dedicated to nearest neighbor search, for instance we have [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl), [`RegionTrees.jl`](https://github.com/rdeits/RegionTrees.jl), and [`JuliaNeighbors`](https://github.com/JuliaNeighbors) implement search structures like [kd-trees](https://en.wikipedia.org/wiki/K-d_tree), [ball trees](https://en.wikipedia.org/wiki/Ball_tree), [quadtrees](https://en.wikipedia.org/wiki/Quadtree), [octrees](https://en.wikipedia.org/wiki/Octree), [bk-trees](https://en.wikipedia.org/wiki/BK-tree), [vp-tree](https://en.wikipedia.org/wiki/Vantage-point_tree) and other multidimensional and metric structures. These structures work quite well for low dimensional data since they are designed to solve exact similarity queries.

There exist several packages performing approximate similarity search, like [`Rayuela.jl`](https://github.com/una-dinosauria/Rayuela.jl) using product quantization schemes, the wrapper for the [`FAISS`](https://faiss.ai/) library [`Faiss.jl`](https://github.com/zsz00/Faiss.jl). The FAISS library provides high-performance implementations of product quantization schemes and locality-sensitive hashing schemes, along with an industrial-strength implementation of the [`HNSW`](https://github.com/nmslib/hnswlib) index. The [`NearestNeighborDescent.jl`](https://github.com/dillondaudert/NearestNeighborDescent.jl) implements the search algorithm behind [`pynndescent`](https://pynndescent.readthedocs.io/en/latest/?badge=latest).

The `SimilaritySearch.jl` package tries to enrich the ecosystem with search structures and algorithms designed to take advantage of multithreading systems and a unique autotuning feature that simplifies its usage for practitioners. These features are succinctly and efficiently implemented due to the Julia programming language dynamism and performance.
Regarding performance characteristics, the construction times are vastly reduced compared to similar approaches without reducing search performance or result quality.

# Installing SimilaritySearch

You may install the package as follows
```julia
] add SimilaritySearch.jl
```

also, you can run the set of tests as follows
```julia
] test SimilaritySearch
```

# Using the library
Please see [examples](https://github.com/sadit/SimilaritySearchDemos). You will find a list of Jupyter and Pluto notebooks, and some scripts that exemplifies its usage.
 
# Contribute
Contributions are welcome. Please fill a pull request for documentating and implementation contributions. For issues, please fill an issue with the necessary information (see below.) If you already have a solution please also provide a pull request.

# Issues
Report issues in the package providing a minimal reproducible example. If the issue is data dependant, please don't forget to provide the necessary data to reproduce it.

## Limitations of `SearchGraph`
The main search structure, the `SearchGraph,` is a graph with several characteristics, many of them induced by the dataset being indexed. Some of its known limitations are related to these characteristics. For instance:

- Metric distances work well; on the other hand, semi-metric should work, but routing capabilities are not yet characterized.
- Even when it performs pretty well compared to alternatives, discrete metrics like Levenshtein distance and others that take few possible values may also get low performances.
- Something similar will happen when there are many near-duplicates (elements that are **pretty** close). In this case, it is necessary to remove near-duplicates and put them in _bags_ associated with some of its near objects.
- Very high dimensional datasets will produce _long-tail_ distributions of the number of edges per vertex. In extreme cases, you must prune large neighborhoods and enrich single-edge paths.

# About the structures and algorithms
The following manuscript describes and benchmarks the `SearchGraph` index (package version `0.6`):

```
@article{tellezscalable,
  title={A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs},
  author={Tellez, Eric S and Ruiz, Guillermo and Chavez, Edgar and Graff, Mario},
  journal={Pattern Analysis and Applications},
  pages={1--15},
  publisher={Springer}
}

``` 

The current algorithm (version `0.8` and `0.9`) is described and benchmarked in the following manuscript:
```

@misc{tellez2022similarity,
      title={Similarity search on neighbor's graphs with automatic Pareto optimal performance and minimum expected quality setups based on hyperparameter optimization}, 
      author={Eric S. Tellez and Guillermo Ruiz},
      year={2022},
      eprint={2201.07917},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

This package is also described in the JOSS paper:

> Eric S. Tellez and Guillermo Ruiz. _`SimilaritySearch.jl`: Autotuned nearest neighbor indexes for Julia_. Journal of Open Source Software [https://doi.org/10.21105/joss.04442](https://doi.org/10.21105/joss.04442).

## About v0.9.X series

The algorithms of this version are the same as v0.8 but break API compatibility:

- Now, it uses the `Polyester` package to handle multithreading instead of Threads.@threads
- Multithreading methods are enabled by default if the process is started with several threads; in v0.8 was the contrary
- `allknn` now preserves self-references to simplify algorithms and improve efficiency (`allknn` in v0.8 removes self-references automatically)

Others:

- Adds function docs and benchmarks
- Adds `SearchGraph` graph pruning methods
- Removes the `timedsearchbatch` function

## About v0.10.X series

It makes easy to adjust the `SearchGraph` structure to different workloads and applications. For instance,
- More control for construction parameters
- Loading and saving
- Refactors search API to be consistent across structs

Please refer to <https://github.com/sadit/SimilaritySearch.jl/blob/main/test/testsearchgraph.jl> for working examples. The demos page will be updated soon.
Nonetheless, some minor (but breaking) changes in the API 
