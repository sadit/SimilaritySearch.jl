[![Build Status](https://travis-ci.org/sadit/SimilaritySearch.jl.svg?branch=master)](https://travis-ci.org/sadit/SimilaritySearch.jl)
[![codecov](https://codecov.io/gh/sadit/SimilaritySearch.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sadit/SimilaritySearch.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/SimilaritySearch.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/SimilaritySearch.jl?branch=master)

# SimilaritySearch.jl


SimilaritySearch.jl is a library for approximate nearest neighbors.


# Installing SimilaritySearch


You may install the package as follows
```bash
julia -e 'using Pkg; pkg"add https://github.com/sadit/SimilaritySearch.jl"'
```
also, you can run the set of tests as fol
```bash
julia -e 'using Pkg; pkg"test SimilaritySearch"'
```

# Indexing and searching
A simple exhaustive search can be implemented as follows:

```julia
julia> using SimilaritySearch
julia> db = [rand(8) for i in 1:100_000]
julia> seqindex = fit(Sequential, db)  # construction
julia> search(seqindex, l2_distance, rand(8), KnnResult(3))  # searching 3-nn for the random vector rand(8)
KnnResult{Int64}(3, Item{Int64}[Item{Int64}(83265, 0.198482), Item{Int64}(44113, 0.219748), Item{Int64}(38506, 0.254233)])
```

`SimilarySearch.jl` supports different kinds of indexes and distance functions. For example, you can create a different index with Manhattan distance as follows
```julia
julia> using SimilaritySearch.Graph
julia> graph = fit(SearchGraph, l1_distance, db)
julia> search(seqindex, l2_distance, rand(8), KnnResult(3))
KnnResult{Int64}(3, Item{Int64}[Item{Int64}(48881, 0.200722), Item{Int64}(56933, 0.224531), Item{Int64}(21200, 0.234252)])
```

Please note that `fit`ing a `SearchGraph` may seems that it pauses for some moments, this is because this kind of methods are designed to compute the best parameters online. We aim that this strategy reduces the complexity of using a searching method, since it tries to achieve the better performance for the given initial configuration.

The package implements several distances, as the following ones:
- [Minkowski family](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/distances/vectors.jl)
  - `l1_distance` also known as Manhattan distance
  - `l2_distance` a.k.a Euclidean distance 
  - squared_l2_distance (not metric)
  - `linf_distance` ($L_∞$) a.k.a. Chebyshev distance
  - and a factory for generic `p` values to define the Minkowski family of distances.
- [Angle distance](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/distances/cos.jl)
  - `angle_distance`
  - `cosine_distance` (not metric, but faster than angle's distance)
  - please not that these functions suppose that your vectors are normalized (it also provides the convenient `normalize!` functions)
- [Binary hamming distance](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/distances/bits.jl)
- [String distances](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/distances/strings.jl)
  - `common_prefix_distance` (not metric)
  - `generic_levenshtein` (with variable costs)
  - `hamming_distance`
  - `levenshtein_distance`
  - `lcs_distance`
- [Set distances](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/distances/sets.jl); here sets are represented as ordered lists (arrays) of integers (in fact, ordered items)
  - `jaccard_distance`
  - `dice_distance` (not metric)
  - `intersection_distance` (not metric)

Please note that you can implement your own distance function and pass to any method, so you can support for any kind of object.

The package is designed to work with approximate indexes, that is, those search methods that are allowed to have false positives and false negatives. However, it supports the following exact methods:
- [Sequential](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/indexes/seq.jl) or exhaustive search
- [LAESA](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/indexes/laesa.jl) or pivot table; it supports different [pivot selection strategies](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/indexes/pivotselectiontables.jl)
- [Kvp](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/knr/kvp.jl) or K vantage points

`SimilaritySearch.jl` implements tghe following approximate methods:
- [Knr](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/knr/knr.jl) K nearest references implemented over an uncompressed inverted index.
- [SearchGraph](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/graph/graph.jl). This is the main method supporting multiple search algorithms for searching and for neighborhood computation (critical for the construction procedure).
  - [IHC](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/graph/ihc.jl). search method inspired in _iterated hill climbing_.
  - [TIHC](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/graph/tihc.jl). search method inspired in a mix of _random search_ and _iterated hill climbing_.
  - [BeamSearch](https://github.com/sadit/SimilaritySearch.jl/blob/master/src/graph/beamsearch.jl). search method inspired in beam search with initial random search.
  
A number of neighborhood computations are also [available](https://github.com/sadit/SimilaritySearch.jl/tree/master/src/graph/neighborhood).
  
**TODO: cite related papers**


# Benchmarking

The package has a benchmarking procedure that can be used to compare different searching methods; this is the same procedure used by internal optimizers

```julia
julia> using SimilaritySearch, SimilaritySearch.Graph, SimilaritySearch.SimilarReferences
julia> db = [rand(Float32, 8) for i in 1:100_000]
julia> queries = [rand(Float32, 8) for i in 1:1_000]
julia> perf = Performance(l2_distance, db, queries, queries_from_db=false, expected_k=10)
julia> seq = fit(Sequential, db)
julia> knr = fit(Knr, l2_distance, db, numrefs=1000, k=3)
julia> graph = fit(SearchGraph, l2_distance, db)
julia> P = (seq=probe(perf, seq, l2_distance), knr=probe(perf, knr, l2_distance), graph=probe(perf, graph, l2_distance))
julia> M = Array{Any}(undef, 4, 5)
julia> M[1, :] .= ["index", "distances_sum", "evaluations_ratio", "queries_by_second", "recall"]
julia> for (i, p) in zip(
            ["seq", "knr", "graph"],
            [p.distances_sum/P[1].distances_sum for p in P],
            [p.evaluations/P[1].evaluations for p in P],
            [1/p.seconds for p in P],
            [p.recall for p in P]) |> enumerate
       M[i+1, :] .= p
       end

julia> M
4×5 Array{Any,2}:
 "index"   "distances_sum"   "evaluations_ratio"       "queries_by_second"   "recall"
 "seq"    1.0               1.0                     199.328                 1.0      
 "knr"    1.01003           0.0197858              2438.6                   0.9084   
 "graph"  1.01298           0.00281581            11273.1                   0.87 
```

You may achieve better performance using different parameters at construction time;
however, this is not always preferred. [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl) also optimizes already built indexes

```julia
julia> optimize!(knr, l2_distance, recall=0.7, k=10)
julia> optimize!(graph, l2_distance, recall=0.7, k=10)
julia> P = (seq=probe(perf, seq, l2_distance), knr=probe(perf, knr, l2_distance), graph=probe(perf, graph, l2_distance))
julia> M = Array{Any}(undef, 4, 5)
julia> M[1, :] .= ["index", "distances_sum", "evaluations_ratio", "queries_by_second", "recall"]
julia> for (i, p) in zip(
            ["seq", "knr", "graph"],
            [p.distances_sum/P[1].distances_sum for p in P],
            [p.evaluations/P[1].evaluations for p in P],
            [1/p.seconds for p in P],
            [p.recall for p in P]) |> enumerate
       M[i+1, :] .= p
       end

julia> M
4×5 Array{Any,2}:
 "index"   "distances_sum"   "evaluations_ratio"      "queries_by_second"   "recall"
 "seq"    1.0               1.0                    200.575                 1.0      
 "knr"    1.00066           0.0341043             1582.46                  0.9905   
 "graph"  1.0009            0.00492749            6261.43                  0.9872   

```

