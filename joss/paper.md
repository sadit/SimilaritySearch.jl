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
Similarity search algorithms are fundamental tools for many computer science and data analysis methods. For instance, they are among the underlying machinery behind efficient information retrieval systems [@sparse-dense-text-retrieval]; they allow performing fast clustering analysis on large datasets [@pmlr-v157-weng21a; jayaram2019diskann; @sisap2020kmeans]. Another outstanding example is how they can speedup the constructions of all $k$ nearest neighbor graphs, that are the basic input of non-linear dimensional reduction methods that it is a popular way to visualize complex data [@umap2018; @trimap2019; van2008visualizing; @lee2007nonlinear;], among other uses. The number of potential applications is also increasing as the number of new representations and problems being solved by deep learning.

## The $k$ nearest neighbor problem
Given a metric dataset, $S \subseteq U$ and a metric distance function $d$, defined for any pair of elements in $U$, 
the $k$ nearest neighbor search of $q$ consists on finding the subset $R$ that minimize $\sum_{u \in R} d(q, u)$ for all possible subsets of size $k$, i.e., $R \subset S$ and $|R| = k$. Elements in $U$ are typically vectors but can have any data representation as long as the metric distance support it.


The problem can be solved easily with an exhaustive evaluation of all possible results $d(u_1, q), \cdots, d(u_n, q)$ (that is, for all $u_i \in S$) and then select those $k$ items $\{u_i\}$ with the least distance to $q$. This solution is impractical when $n$ is large, or the expected number of queries is high, or the intrinsic dimension of the dataset is also high \cite{rub}. In these cases, it is necessary to create a data structure that preprocess the dataset and reduce the cost of solving queries, it is often called an \textit{index}. More even, sometimes we can loose the ability of retrieving the exact solution to gain speed, clearly, the approximation quality becomes a major concern and these approximate methods require a lot of knowledge to trade speed retrieval process also kept high the solution's quality. Additionally, the amount of memory used by the index and the construction time are also concerns whenever $n$ is big.

# Main features of `SimilaritySearch`

The core method, the `SearchGraph` struct is an approximate method that is designed to trade effectively between speed and quality, it has an integrated auto-tuning feature that free users of almost any setup and manual model selection. More detailed, in a single construction, the incremental construction adjusts the index parameters to achieve the desired performance. @tellez2021scalable

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

The precise definitions of these functions and the full set of functions and structures can be found in is documentation.^[ [https://sadit.github.io/SimilaritySearch.jl/](https://sadit.github.io/SimilaritySearch.jl/) ]


# Installation and usage
The package is registered in the general Julia registry and it is available via its integrated package manager:
```julia
using Pkg
Pkg.add("SimilaritySearch")
```

Example:

```julia
...
```


You can find several examples and notebooks (Pluto and Jupyter) in ^[ [https://github.com/sadit/SimilaritySearchDemos)(https://github.com/sadit/SimilaritySearchDemos) ].


# Acknowledgements

# References