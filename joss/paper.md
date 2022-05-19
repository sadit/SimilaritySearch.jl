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
 - name: Consejo Nacional de Ciencia y Tecnología, México.
   index: 1
 - name: INFOTEC Centro de Investigación e Innovación en Tecnologías de la Información y Comunicación, México.
   index: 2
 - name: CentroGEO Centro de Investigación en Ciencias de Información Geoespacial, México.
   index: 3
date: 20 May 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

This manuscript describes the `SimilaritySearch.jl` Julia's package (MIT licensed) that provides algorithms to efficiently retrieve $k$ nearest neighbors from a metric dataset and other related problems with no knowledge of the underlying algorithms since our main structure, the `SearchGraph,` has autotuning capabilities. Its algorithms are designed to work in main memory and take advantage of multithreading systems in most of its primary operations.

# Statement of need
Similarity search algorithms are fundamental tools for many computer science and data analysis methods. For instance, they are among the underlying machinery behind efficient information retrieval systems [@witten1999managing,@sparse-dense-text-retrieval]; they allow fast clustering analysis on large datasets [@pmlr-v157-weng21a; @jayaram2019diskann; @sisap2020kmeans]. Another outstanding example is how they can speed up the constructions of all $k$ nearest neighbor graphs, which are the input of non-linear dimensional reduction methods. It is a popular way to visualize complex data [@umap2018; @trimap2019; @van2008visualizing; @lee2007nonlinear;], among other use cases. The number of potential applications is also increasing as the number of problems solved by deep learning methods proliferates, i.e., many deep learning internal representations are direct input for similarity search.


## The $k$ nearest neighbor problem
Given a metric dataset, $S \subseteq U$ and a metric distance function $d$, defined for any pair of elements in $U$, 
the $k$ nearest neighbor search of $q$ consists on finding the subset $R$ that minimize $\sum_{u \in R} d(q, u)$ for all possible subsets of size $k$, i.e., $R \subset S$ and $|R| = k$. 
The problem can be solved easily with an exhaustive evaluation, but this solution is impractical when the number of expected queries is large or for high-dimensional datasets. When the dataset can be preprocessed, it is possible to overcome these difficulties by creating an \textit{index}, i.e., a data structure to solve similarity queries efficiently.

Our `SearchGraph` is based on the Navigable Small World (NSW) graph index [@malkov2018efficient] using a different search algorithm based on the well-known beam search meta-heuristic, smaller node degrees based on Spatial Access Trees [@navarro2002searching], and auto-tuned capabilities. The details are studied in [@simsearch2022; @tellez2021scalable; @ruiz2015finding]. The package solves other related problems using these indexes as internal machinery.

## Alternatives
@malkov2014approximate add a hierarchical structure to the NSW to create the Hierarchical NSW (HNSW) search structure. This index is a central component of popular libraries^[https://github.com/nmslib/hnswlib; https://github.com/nmslib/nmslib; https://github.com/facebookresearch/faiss] and has a significant acceptance in the community. @nndescent11 introduces the NN Descent method, which uses the graph of neighbors as index structure; it is the machinery behind PyNNDescent^[<https://github.com/lmcinnes/pynndescent>], which is behind the fast computation of UMAP non-linear low dimensional projection.^[<https://github.com/lmcinnes/umap>]
@scann2020 introduced the _scann_ index for inner product-based metrics; it is fast and accurate and implemented in a well-maintained library.^[<https://github.com/google-research/google-research/tree/master/scann>]

# Main features of `SimilaritySearch`

The `SearchGraph` struct is an approximate method designed to trade effectively between speed and quality. It has an integrated autotuning feature that almost free users of any setup and manual model selection. In a single pass, the incremental construction adjusts the index parameters to achieve the desired performance, whether bi-objective (Pareto optimal for search speed and quality) or a minimum quality. This search structure is described in [@simsearch2022], which uses the `SimilaritySearch.jl` package as implementation (0.8 version series). Older versions of the package are benchmarked in [@tellez2021scalable].

The package provides the following indexes:

- `ParallelExhaustiveSearch`: A brute force search index where each query is solved using all available threads.
- `ExhaustiveSearch`: A brute force search index, each query is solved in a single thread.
- `SearchGraph`: An approximate search index with parallel and online autotuned construction.

The main set of functions are:

- `search`: Solves a single query.
- `searchbatch`: Solves a set of queries.
- `allknn`: Computes the $k$ nearest neighbors for all elements in an index.
- `closestpair`: Computes the closest pair in a metric dataset.
- `neardup`: Removes near-duplicates from a metric dataset.

The complete set of functions and structures are detailed in the documentation.^[<https://sadit.github.io/SimilaritySearch.jl/>]

# Installation and usage
The package is registered in the general Julia registry, and it is available via its integrated package manager:
```julia
using Pkg
Pkg.add("SimilaritySearch")
```

The package exports several functions and indexes for solving similarity search queries, as mentioned above. For instance, the set of 60k-10k train set partition of hand-written digits MNIST dataset [@lecun1998gradient], using the `MLDatasets` (v0.6.0) package for this matter, is used to exemplify the use of the `SimilaritySearch.jl` (v0.8.18) Julia package.

~~~~ {#example .julia .numberLines startFrom="1"}
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
  index!(G; parallel_block=256)
  id, dist = searchbatch(G, queries, k; parallel=true)
  point1, point2, mindist = closestpair(G; parallel=true)
  idAll, distAll = allknn(G, k; parallel=true)
end

example()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function `example` loads the data (line 12), create the index (line 14) and then finds all $k$ nearest neighbors of the test partition in the indexed partition as a batch of queries (line 15). The same index is used to compute the closest pair of points in the train partition (line 16) and finally compute all $k$ nearest neighbors on the train partition (line 17). All these operations use all the available threads to the `julia` process.


We ran this example in an Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz workstation with 256GiB RAM using GNU/Linux CentOS 8. Our system has 32 cores with hyperthreading activated (64 threads). We used the v0.8.18 version of our package and julia 1.7.2. Table \ref{tab/performance} compares the running times with those achieved with the brute force algorithm (replacing lines 13-14 with `ExhaustiveSearch(; dist, db)`). We also compared an optimized version of our index resulting from calling `optimize!(G, MinRecall(0.95))` after the `index!` function call.
----------------------------------------------------------------------------------------
                 method    build    opt.         search   closestpair    allknn       mem.   recall
 -––––––––––––––––––––––   ––––––   ––––––   ––––––––––   –––––––––––   –––––––   ––––––––   ––––––
        ExhaustiveSearch      0.0      0.0       3.5612       22.1781   21.6492   179.4434   1.0000 \hline
            ParetoRecall   1.5959      0.0       0.1352        0.2709    0.6423   181.5502   0.8204
          MinRecall(0.9)    -       0.2572       0.1796        0.3527    0.9209      -       0.8912
         MinRecall(0.95)    -       0.4125       0.4708        0.8333    2.6709      -       0.9635
          MinRecall(0.6)    -       0.1190       0.0588        0.2207    0.2618      -       0.5914 \hline
  Hamming MinRecall(0.9)   1.1323   0.0729       0.0438        0.2855    0.2175     8.4332   0.7053
----------------------------------------------------------------------------------------

Table: Performance comparison of running several similarity search operations on MNIST dataset in our 32-core workstation. Operations taking with small-time costs is desirable, while higher throughput $q/s$ and high recall scores (close to 1) are also desirable. \label{tab/performance}

The reported recall score is the macro averaged recall of the 60k $k$ nearest neighbors sets computed by the `allknn` operation. The individual recall is computed as ${\# \text{ of actual } k \text{ nearest neighbors retrieved}}/{k}$. The set of actual $k$ nearest neighbors is the intersection of the set of $k$ nearest neighbors computed by the brute force method and the index being compared, and it takes values between 0 and 1. This score is computed easily with the `macrorecall` function also implemented in `SimilaritySearch.jl`.

Our implementations produce complete results when _exact_ indexes are used and will produce approximate results when approximate indexes are used.
More examples and notebooks (Pluto and Jupyter) are available in the sister repository <https://github.com/sadit/SimilaritySearchDemos>.

# Acknowledgements
This research uses some of the computing infrastructure of the _Laboratorio de GeoInteligencia Territorial_ at _CentroGEO Centro de Investigación en Ciencias de Información Geoespacial_, Aguascalientes, México.

# References