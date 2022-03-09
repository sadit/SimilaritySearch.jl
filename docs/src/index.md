```@meta
CurrentModule = SimilaritySearch
```

# SimilaritySearch.jl

SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for `SearchGraph`, a fast and flexible search index.

The following manuscript describes and benchmarks version `0.6`:

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
Please see [examples](https://github.com/sadit/SimilaritySearchDemos). You will find a list of Jupuyter and Pluto notebooks, and some scripts that exemplifies its usage.
 
