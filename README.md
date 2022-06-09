[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/SimilaritySearch.jl/dev)
[![Build Status](https://github.com/sadit/SimilaritySearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/SimilaritySearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/SimilaritySearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/SimilaritySearch.jl)


# SimilaritySearch.jl


SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for `SearchGraph`, a fast and flexible search index using any metric function. It is designed to support multithreading in most of its main functions and structures.

The package provides the following indexes:

- `ParallelExhaustiveSearch`: An brute force search index where each query is solved using all available threads.
- `ExhaustiveSearch`: A brute force search index, each query is solved using a single thread.
- `SearchGraph`: An approximate search index with parallel construction.

The main set of functions are:

- `search`: Solves a single query.
- `searchbatch`: Solves a set of queries.
- `allknn`: Computes the $k$ nearest neighbors for all elements in an index.
- `neardup`: Removes a near duplicates from a metric dataset.
- `closestpair`: Computes the closest pair in a metric dataset.

The precise definitions of these functions and the full set of functions and structures can be found in the [documentation](https://sadit.github.io/SimilaritySearch.jl/dev).

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
The main search structucture, the `SearchGraph` is a graph with several characteristics, many of them, induced by the dataset being indexed. Some of its known limitations are related with these characteristics. For instance:

- Metric distances are known to work well, in the other hand, semi-metric should work but routing capabilities are not yet characterized.
- Even when it performs pretty well as compared with alternatives, discrete metrics like Levenshtein distance and others that take few possible values may also get low performances.
- Something similar will happen when there exists a lot of near duplicates (elements that are **pretty** close). In this case, it is necessary to remove near duplicates and put them in _bags_ associated to some of its near object.
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

The current algorithm (version `0.8`) is described and benchmarked in the following manuscript:
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
