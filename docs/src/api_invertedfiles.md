```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```


# Inverted files and sparse data

## Inverted files (`InvertedFiles` submodule)

Inverted file index data structures and context for sparse vectors, MIPS, and set search.

```@docs
InvertedFiles.AbstractInvertedFile
InvertedFiles.InvertedFile
InvertedFiles.DictInvertedFile
InvertedFiles.search_invfile
InvertedFiles.select_posting_lists
InvertedFiles.has_exact_fastpath
InvertedFiles.identiterator
InvertedFiles.sort_postinglist!
InvertedFiles.InvertedFileContext
InvertedFiles.getcontext
InvertedFiles.set_distance_evaluate
```

## Posting list intersections (`Intersections` submodule)

Algorithms for set and posting list intersections.

```@docs
Intersections.bk!
Intersections.bkt!
Intersections.umerge!
Intersections.xmerge!
Intersections.svs
Intersections.binarysearch
Intersections.doublingsearch
Intersections.doublingsearchrev
Intersections.seqsearch
Intersections.seqsearchrev
Intersections.imerge2!
```

## Spherical embedding for MIPS (`Special.Spherical` submodule)

Turns Maximum Inner Product Search into ordinary nearest-neighbor search (Neyshabur &
Srebro's asymmetric spherical embedding), for dense and sparse vectors alike.

```@docs
Special.Spherical
Special.Spherical.SphericalEmbedding
Special.Spherical.outdim
Special.Spherical.indim
Special.Spherical.transform
Special.Spherical.transform!
Special.Spherical.transform_query
Special.Spherical.transform_query!
```

## Sparse vector support (`Special.Sparse` submodule)

A sparse matrix view tailored for distance evaluations, replacing Base's `SparseVector`
with an explicit dimension-tracking read-only wrapper `SparseVecView`.

```@docs
Special.Sparse.sparsedot
Special.Sparse.centroid
```
