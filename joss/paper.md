---
title: '`SimilaritySearch.jl`: Autotuned nearest neighbor indexes for Julia'
tags:
  - Autotuned similarity search indexes
  - K nearest neighbor search
  - All KNN queries
  - Near duplicates detection
  - Closest pair queries
authors:
  - name: Eric S. Tellez
    orcid: 0000-0001-5804-9868
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Guillermo Ruiz
    orcid: 0000-0001-7422-7011
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

This manuscript describes the MIT-licensed Julia [@bezanson2017julia] package `SimilaritySearch.jl` that provides algorithms to efficiently retrieve $k$ nearest neighbors from a metric dataset and other related problems with no knowledge of the underlying algorithms, since our main structure, the `SearchGraph,` has autotuning capabilities. The package is designed to work in main memory and takes advantage of multithreading systems in most of its primary operations.

# Statement of need
Similarity search algorithms are fundamental tools for many computer science and data analysis methods. For instance, they are among the underlying machinery behind efficient information retrieval systems [@witten1999managing; @sparse-dense-text-retrieval], and they allow fast clustering analysis on large datasets [@pmlr-v157-weng21a; @jayaram2019diskann; @sisap2020kmeans]. Another example is how they can speed up the construction of all $k$ nearest neighbor graphs, which are the input of non-linear dimensional reduction methods that are popularly used to visualize complex data [@umap2018; @trimap2019; @van2008visualizing; @lee2007nonlinear]. The number of potential applications is also increasing as the number of problems solved by deep learning methods proliferates, i.e., many deep learning internal representations are direct input for similarity search.

## The $k$ nearest neighbor problem
Given a metric dataset, $S \subseteq U$ and a metric distance function $d$, defined for any pair of elements in $U$, the $k$ nearest neighbor search of $q$ consists of finding the subset $R$ that minimize $\sum_{u \in R} d(q, u)$ for all possible subsets of size $k$, i.e., $R \subset S$ and $|R| = k$.
The problem can be solved easily with an exhaustive evaluation, but this solution is impractical when the number of expected queries is large or for high-dimensional datasets. When the dataset can be preprocessed, it is possible to overcome these difficulties by creating an \textit{index}, i.e., a data structure to solve similarity queries efficiently. Depending on the dimensionality and size of the dataset, it could be necessary to trade speed for quality,^[The quality is often measured as the `recall,` which is as a proportion of how many relevant results were found in a search; our package contains a function `macrorecall` that computes the average of this score for a set of query results.] traditional methods leave this optimization to the user. Our approach has automated functions that simplify this task.

Our `SearchGraph` is based on the Navigable Small World (NSW) graph index [@malkov2018efficient] using a different search algorithm based on the well-known beam search meta-heuristic, smaller node degrees based on Spatial Access Trees [@navarro2002searching], and autotuned capabilities. The details are provided in [@simsearch2022; @tellez2021scalable; @ruiz2015finding].

## Alternatives
@malkov2014approximate add a hierarchical structure to the NSW to create the Hierarchical NSW (HNSW) search structure. This index is a central component of the [`hnswlib`](https://github.com/nmslib/hnswlib) and the [`nmslib`](https://github.com/nmslib/nmslib) libraries. Along with the HNSW, the [`faiss`](https://github.com/facebookresearch/faiss) library also provides a broad set of efficient implementations of metric, hashing, and product quantization indexes. @nndescent11 introduce the NN Descent method, which uses the graph of neighbors as index structure; it is the machinery behind [`PyNNDescent`](https://github.com/lmcinnes/pynndescent), which is behind the fast computation of UMAP non-linear low dimensional projection.^[<https://github.com/lmcinnes/umap>.]
@scann2020 introduce the _SCANN_ index for inner product-based metrics and Euclidean distance, available at the [SCANN repository](https://github.com/google-research/google-research/tree/master/scann) based on hashing.

Currently, there are some packages dedicated to nearest neighbor search, for instance, [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl), [`Rayuela.jl`](https://github.com/una-dinosauria/Rayuela.jl), [`HNSW.jl`](https://github.com/JuliaNeighbors/HNSW.jl), and a wrapper for the FAISS library, [`Faiss.jl`](https://github.com/zsz00/Faiss.jl), among other efforts.


# Main features of `SimilaritySearch`

The `SearchGraph` struct is an approximate method designed to trade effectively between speed and quality. It has an integrated autotuning feature that almost free the users of any setup and manual model selection. In a single pass, the incremental construction adjusts the index parameters to achieve the desired performance, optimizing both search speed and quality or a minimum quality. This search structure is described in [@simsearch2022], which uses the `SimilaritySearch.jl` package as implementation (0.9 version series). Previous versions of the package are benchmarked in [@tellez2021scalable].

The main set of functions are:

- `search`: Solves a single query.
- `searchbatch`: Solves a set of queries.
- `allknn`: Computes the $k$ nearest neighbors for all elements in an index.
- `closestpair`: Computes the closest pair in a metric dataset.
- `neardup`: Removes near-duplicates from a metric dataset.

Note that our implementations produce complete results with _exact_ indexes and will produce approximate results when _approximate_ indexes are used.

`SimilaritySearch.jl` can be used with any semi-metric, as defined in the package [`Distances.jl`](https://github.com/JuliaStats/Distances.jl). Note that a number of distance functions for vectors, strings, and sets are also available in our package.

The complete set of functions and structures are detailed in the documentation.^[<https://sadit.github.io/SimilaritySearch.jl/>]

# Installation
The package is available in the Julia's integrated package manager:
```julia
using Pkg
Pkg.add("SimilaritySearch")
```

# A brief example and a comparison with alternatives
As an example, we used the set of 70k hand-written digits MNIST dataset [@lecun1998gradient] (using the traditional partition scheme of 60k objects for indexing and 10k as queries). We use the [`MLDatasets.jl`](https://github.com/JuliaML/MLDatasets.jl) package for this matter (v0.6); each 28x28 image is loaded as a 784-dimensional vector using 32-bit floating-point numbers. We select the squared Euclidean distance as the metric.

~~~~ {#example .julia .numberLines startFrom="1"}
using SimilaritySearch, MLDatasets 

function load_data()
  train, test = MNIST(split = :train), MNIST(split = :test)
  (w, h, n), m = size(train.features), size(test.features, 3)
  db = MatrixDatabase(reshape(train.features, w * h, n))
  queries = MatrixDatabase(reshape(test.features, w * h, m))
  db, queries
end

function example(k, dist = SqL2Distance())
  db, queries = load_data()
  G = SearchGraph(; dist, db)
  index!(G; parallel_block = 512)
  id, dist = searchbatch(G, queries, k)
  point1, point2, mindist = closestpair(G)
  idAll, distAll = allknn(G, k)
end

example(32)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function `example` loads the data (line 12), creates the index (line 14), and then finds all $k$ nearest neighbors of the test in the indexed partition as a batch of queries (line 15). The same index is used to compute the closest pair of points in the train partition (line 16) and compute all $k$ nearest neighbors on the train partition (line 17) for $k=32$.

For this, we used an Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz workstation with 256 GiB RAM using GNU/Linux CentOS 8. Our system has 32 cores (64 threads), and we use all threads in all tested systems. For instance, we used `SimilaritySearch.jl` v0.9.3 and `Julia` 1.7.2. Table \ref{tabperformance} compares the running times of `SearchGraph` (SG). We consider different autotuned versions calling `optimize!(G, MinRecall(r))` after the `index!` function call, for different expected recall scores, it defaults to `ParetoRecall`. We also compare with a parallel brute-force algorithm (replacing lines 13-14 with `ExhaustiveSearch(; dist, db)`).

\begin{table}[!ht]

\caption{Performance comparison of running several similarity methods on the MNIST dataset. Smaller time costs and memory are desirable while high recall scores (close to 1) are better.}
\label{tabperformance}
\resizebox{\textwidth}{!}{
\begin{tabular}{cccc cccc}
\hline
method                      & build    & opt.     & \texttt{searchbatch}  & \texttt{closestpair}  & \texttt{allknn}  & mem.   & \texttt{allknn} \\
                            & cost (s) & cost (s) &  cost (s)             & cost (s)              & cost (s)         & (MB)   &  recall \\ \hline
ExhaustiveSearch            &  0.0     & 0.0      &   3.56                &  22.18                & 21.65            & 179.44 &  1.00   \\ \hline
SG ParetoRecall             &  0.91    & 0.0      &   0.10                &   0.29                &  0.41            & 182.22 &  0.78   \\
SG \texttt{MinRecall(0.6)}  &  ''      & 0.10     &   0.04                &   0.11                &  0.19            &  ''    &  0.66   \\
SG \texttt{MinRecall(0.9)}  &  ''      & 0.12     &   0.13                &   0.46                &  0.61            &  ''    &  0.86   \\
SG \texttt{MinRecall(0.95)} &  ''      & 0.23     &   0.15                &   0.55                &  0.75            &  ''    &  0.93   \\ \hline
SCANN                       & 25.11    &  -       &     -                 &     -                 &  2.14            & 201.95 &  1.00   \\
HNSW (FAISS)                &  1.91    &  -       &     -                 &     -                 &  1.99            & 195.02 &  0.99   \\
PyNNDescent                 & 45.09    &  -       &     -                 &     -                 &  9.94            & 430.42 &  0.99   \\
\hline
\end{tabular}
}
\end{table}


## Comparison with alternatives
We also indexed and searched for all $k$ nearest neighbors using the default values for the HNSW, PyNNDescent, and SCANN nearest neighbor search indexes. All these operations were computed using all available threads. Note that high recall scores indicate that the default parameters can be adjusted to improve search times; nonetheless, optimizing parameters also imply using a model selection procedure that requires more computational resources and knowledge about the packages and methods.
Our `SearchGraph` (SG) method performs this procedure in a single pass and without extra effort by the user. Note that we run several optimizations that use the same index and spend a small amount of time effectively trading between quality and speed; this also works for larger and high-dimensional datasets as benchmarked in @simsearch2022.
Finally, short-lived tasks like computing all $k$ nearest neighbors for non-linear dimensional reductions (e.g., data visualization) also require low build costs; therefore, a complete model selection is prohibitive, especially for large datasets.

# Final notes
`SimilaritySearch.jl` provides a metric-agnostic alternative for similarity search in high-dimensional datasets. Additionally, our autotuning feature is a milestone in the nearest neighbor community since it makes the technology more accessible for users without profound knowledge in the field.
More examples and notebooks (Pluto and Jupyter) are available in the sibling repository <https://github.com/sadit/SimilaritySearchDemos>.

# Acknowledgements
The authors would like to thank the reviewers and the editor for their valuable time; their suggestions improved the quality of this manuscript.
This research used the computing infrastructure of the _Laboratorio de GeoInteligencia Territorial_ at _CentroGEO Centro de Investigación en Ciencias de Información Geoespacial_, Aguascalientes, México. 

# References
