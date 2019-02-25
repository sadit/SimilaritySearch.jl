[![Build Status](https://travis-ci.org/sadit/SimilaritySearch.jl.svg?branch=master)](https://travis-ci.org/sadit/SimilaritySearch.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/SimilaritySearch.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/SimilaritySearch.jl?branch=master)

# SimilaritySearch.jl


SimilaritySearch.jl is a library for approximate nearest neighbors, performing quite good as compared with the state of the art techniques.

Our aim is to develop a fast and easy to use library for nearest neighbor search.


# Cloning and testing

First clone the repository
```bash
git clone https://github.com/sadit/SimilaritySearch.jl
```

To test the package you run the following inside the package directory
```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'
```