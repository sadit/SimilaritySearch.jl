---
title: '`SimilaritySearch.jl`: Fast nearest neighbor searches in Julia'
tags:
  - Julia
  - Autotuned similarity search indexes
  - K nearest neighbor search
  - All KNN queries
  - Near duplicates detection
  - Closest pair queries
authors:
  - name: Eric S. Tellez^[Co-first author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0001-5804-9868
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Guillermo Ruiz^[Co-first author] # note this makes a footnote saying 'Co-first author'
    affiliation: "1, 3"
affiliations:
 - name: Consejo Nacional de Ciencia y Tecnología, México
   index: 1
 - name: INFOTEC Centro de Investigación e Innovación en Tecnologías de la Información y Comunicación, México
   index: 2
 - name: CentroGEO Centro de Investigación en Ciencias de Información Geoespacial, México
   index: 3
date: 20 May 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

This manuscript describes the `SimilaritySearch.jl` Julia's package that provides algorithms to retrieve efficiently $k$ nearest neighbors from a metric dataset and other related problems. Its algorithms are designed to work in main memory and take advantage of multithreading systems in most of its main operations.

# Statement of need
Similarity search algorithms are fundamental tools for many computer science and data analysis methods. For instance, they are among the underlying machinery behind efficient information retrieval systems [@sparse-dense-text-retrieval]; they allow performing fast clustering analysis on large datasets [@pmlr-v157-weng21a; @jayaram2019diskann; @sisap2020kmeans]. Another outstanding example is how they can speedup the constructions of all $k$ nearest neighbor graphs, that are the basic input of non-linear dimensional reduction methods that it is a popular way to visualize complex data [@umap2018; @trimap2019; @van2008visualizing; @lee2007nonlinear;], among other uses. The number of potential applications is also increasing as the number of new representations and problems being solved by deep learning.

## The $k$ nearest neighbor problem
Given a metric dataset, $S \subseteq U$ and a metric distance function $d$, defined for any pair of elements in $U$, 
the $k$ nearest neighbor search of $q$ consists on finding the subset $R$ that minimize $\sum_{u \in R} d(q, u)$ for all possible subsets of size $k$, i.e., $R \subset S$ and $|R| = k$. Elements in $U$ are typically vectors but can have any data representation as long as the metric distance support it.

The problem can be solved easily with an exhaustive evaluation of all possible results $d(u_1, q), \cdots, d(u_n, q)$ (that is, for all $u_i \in S$) and then select those $k$ items $\{u_i\}$ with the least distance to $q$. This solution is impractical when $n$ is large, or the expected number of queries is high, or the intrinsic dimension of the dataset is also high. It is possible to overcome some of the difficulties preprocessing the dataset to create a data structure known as an \textit{index}. 

Our `SearchGraph` is based on the Navigable Small World (NSW) graph index [@malkov2018efficient] using a different search algorithm based on the well-known beam search meta-heuristic and small node degrees based on Spatial Access Trees [@navarro2002searching]. The details are studied in [@ruiz2015finding; @tellez2021scalable], and its auto-tuned capabilities in [@simsearch2022].

## Alternatives
@malkov2014approximate add a hierarchical structure to the NSW to create the Hierarchical NSW (HNSW) search structure. This index is a main component of popular libraries ^[https://github.com/nmslib/hnswlib; https://github.com/nmslib/nmslib; https://github.com/facebookresearch/faiss].@nndescent11 introduce NN Descent method, which uses the graph of neighbors as index structure; it is the machinery behind PyNNDescent^[<https://github.com/lmcinnes/pynndescent>], which is behind fast computation of UMAP non-linear low dimensional projection [<https://github.com/lmcinnes/umap>].
Recently, @scann2020 introduces the _scann_ index for inner product based metrics; it is fast and accurate implemented in a well maintained library.^[<https://github.com/google-research/google-research/tree/master/scann>]

# Main features of `SimilaritySearch`

The `SearchGraph` struct is an approximate method that is designed to trade effectively between speed and quality, it has an integrated auto-tuning feature that almost free users of any setup and manual model selection. More detailed, in a single construction, the incremental construction adjusts the index parameters to achieve the desired performance which can be a be bi-objetive (Pareto optimal for search speed and quality) or a minimum quality. This search structure is described in [@simsearch2022] which uses the `SimilaritySearch.jl` package as implementation (0.8 version series). Older versions of the package are benchmarked in [@tellez2021scalable].

The package provides the following indexes:

- `ParallelExhaustiveSearch`: A brute force search index where each query is solved using all available threads.
- `ExhaustiveSearch`: A brute force search index, each query is solved in a single thread.
- `SearchGraph`: An approximate search index with parallel and online autotuned construction.

The main set of functions are:

- `search`: Solves a single query.
- `searchbatch`: Solves a set of queries, 
- `allknn`: Computes the $k$ nearest neighbors for all elements in an index.
- `closestpair`: Computes the closest pair in a metric dataset.
- `neardup`: Removes a near duplicates from a metric dataset.

The full set of functions and structures are detailed in the documentation.^[<https://sadit.github.io/SimilaritySearch.jl/>]

# Installation and usage
The package is registered in the general Julia registry and it is available via its integrated package manager:
```julia
using Pkg
Pkg.add("SimilaritySearch")
```

After this, you can ran unit testing calling `Pkg.test("SimilaritySearch")`. The package exports a number of functions and indexes for solving similarity search queries.

The set of 60k-10k train set partition of hand-written digits MNIST dataset [@lecun1998gradient], using the `MLDatasets` (v0.6.0) package for this matter, is used for exemplify the use of the `SimilaritySearch.jl` (v0.8.17) Julia package.

~~~~ {#example .julia .numberLines startFrom="1"}
# run julia using `-t auto` in a multithreading system
using SimilaritySearch, MLDatasets 

function load_data()
  train, test = MNIST(split=:train), MNIST(split=:test)
  (w, h, n), m = size(train.features), size(test.features, 3)
  db = MatrixDatabase(reshape(train.features, w * h, n))
  queries = MatrixDatabase(reshape(test.features, w * h, m))
  db, queries
end

function example(k=15, dist=SqL2Distance())
  db, queries = load_data()
  G = SearchGraph(; dist, db)
  index!(G; parallel_block=256) # build the index
  id, dist = searchbatch(G, queries, k; parallel=true)
  point1, point2, mindist = closestpair(G; parallel=true)
  idAll, distAll = allknn(G, k; parallel=true)
end

example()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function `example` loads the data (line 13), creates the index (line 15) and then it finds all $k$ nearest neighbors of the test partition in the indexed partition as a batch of queries (line 16). The same index is used to compute the closest pair of points in the train partition (line 17), and finally computes the all $k$ nearest neighbors on the train partition (line 18). All these operations are called using all the available threads to the `julia` process.


We ran this example in an Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz workstation with 256GiB RAM using GNU/Linux CentOS 8. Our system has 32 cores with hyperthreading activated (64 threads). We used the v0.8.17 version of our package and julia 1.7.2. Table [#tab/performance] compares the running times with those achieved with the brute force algorithm (replacing lines 14-15 with `ExhaustiveSearch(; dist, db)`). We also compared an optimized version of our index resulting from calling `optimize!(G, MinRecall(0.95))` after the `index!` function call.

----------------------------------------------------------------------------------------------
                                 `ExhaustiveSearch`     `SearchGraph`       `SearchGraph` with
 Operation             units                                                 `MinRecall(0.95)`
--------------         ------   -------------------   --------------   ----------------------- 
 construction             $s$                  0.00             5.67                     idem

 `optimize!`              $s$                  0.00             0.00                     0.82

 `searchbatch`          $q/s$              2689.68          36340.06                 23807.14

 `closestpair`            $s$                 22.76             0.31                     1.57 

 `allknn`                 $s$                 21.69             0.37                     2.79

 `allknn`                 $s$                   1.0             0.81                     0.96
 recall score 
---------------------------------------------------------------------------------------------

[#tab/performance]: Table: Performance comparison of running several similarity search operations on MNIST dataset in our 32-core workstation.

The reported recall score is the macro averaged recall of `allknn` operation. The individual recall is computed as $\frac{\# \text{ of real } k \text{ nearest neighbors retrieved}}{k}$ where the set of real $k$ nearest neighbors is the intersection of the set of $k$ nearest neighbors retrieved by the brute force method and the index being computed.


Please note that exact indexes produce exact results when these functions are applied while approximate indexes can produce approximate results.
You can find more examples and notebooks (Pluto and Jupyter) in ^[ [https://github.com/sadit/SimilaritySearchDemos)(https://github.com/sadit/SimilaritySearchDemos) ].


# Acknowledgements

# References