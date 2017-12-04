[![Build Status](https://travis-ci.org/sadit/SimilaritySearch.jl.svg?branch=master)](https://travis-ci.org/sadit/SimilaritySearch.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/SimilaritySearch.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/SimilaritySearch.jl?branch=master)

# A Near Neighbor Search Library


SimilaritySearch.jl is a library for approximate nearest neighbors, performing quite good as compared with the state of the art techniques.

In constrast to other approaches, NNS.jl is not only designed to be fast and powerful but also easy to use.
Basically, you only need to declare what kind of data and distance will be indexed, the desired performance, and SimilaritySearch.jl will try to get a the best tradeoff between quality and performance under the desired algorithm.

SimilaritySearch.jl comes with some simple demonstation tools to help on the For instance, you can try the following

```bash

mkdir dbs && cd dbs
curl -O http://ws.ingeotec.mx/~sadit/datasets/nasa.vecs.gz
curl -O http://ws.ingeotec.mx/~sadit/datasets/nasa.vecs.queries.gz
gzip -d nasa.vecs.gz nasa.vecs.queries.gz
cd ..
```

Here you got a database of ~40K objects (dim. 20), splitted in two parts: a database and a set of queries

```
julia tools/nns.jl create,benchmark '{"db": "dbs/nasa.vecs", "index": "nasa.index", "queries": "dbs/nasa.vecs.queries", "results": "nasa.benchmark", "search_algo": "nsearch", "recall": 0.95}'
```


Take into account that the recall adjustment is computed using a set of random items from the dataset, so the real recall for an unseen query can be different depending on its distribution as compared with the known objects. The query set is never touched for the parameter optimization, for the sake of result soundness; however, real applications should know the distribution of their expected queries.

## Methods

As mentioned, this package is intented to offer the basic functionality for similarity search methods; however, it only provides two kind of similarity search indexes, Sequential and Laesa (pivot table); for more sophisticated similarity search please considere the use of the following packages, based on SimilaritySearch:

- [https://github.com/sadit/SimilarReferences.jl](https://github.com/sadit/SimilarReferences.jl) 
- [https://github.com/sadit/NearNeighborGraph.jl](https://github.com/sadit/NearNeighborGraph.jl)

## 
## Final notes ##
To reach maximum performance, please ensure that Julia has access to the precise instruction set of your architecture

[http://docs.julialang.org/en/latest/devdocs/sysimg/](http://docs.julialang.org/en/latest/devdocs/sysimg/)
