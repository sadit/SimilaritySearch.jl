[![Build Status](https://travis-ci.org/sadit/SimilaritySearch.jl.svg?branch=master)](https://travis-ci.org/sadit/SimilaritySearch.jl)
[![codecov](https://codecov.io/gh/sadit/SimilaritySearch.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sadit/SimilaritySearch.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/SimilaritySearch.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/SimilaritySearch.jl?branch=master)

# SimilaritySearch.jl


SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for `SearchGraph`:

_Tellez, E. S., Ruiz, G., Chavez, E., & Graff, M.A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs. Pattern Analysis and Applications, 1-15._

```
@article{tellezscalable,
  title={A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs},
  author={Tellez, Eric S and Ruiz, Guillermo and Chavez, Edgar and Graff, Mario},
  journal={Pattern Analysis and Applications},
  pages={1--15},
  publisher={Springer}
}
```

# Installing SimilaritySearch


You may install the package as follows
```bash
julia -e 'using Pkg; pkg"add SimilaritySearch.jl"'
```
also, you can run the set of tests as fol
```bash
julia -e 'using Pkg; pkg"test SimilaritySearch"'
```

# Using the library
Please see [examples](https://github.com/sadit/SimilaritySearch.jl/tree/master/examples) directory of this repository. Here you will find a list of Pluto's notebooks that exemplifies its usage.
 
**TODO: cite related papers**


