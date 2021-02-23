```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# KnnResult

SimilaritySearch's core is to solve knn searches; for this matter, it relies on the `KnnResult` struct and its related functions.

```@docs

Item
KnnResult
maxlength
covrad
maxlength
push!(res::KnnResult, id::Integer, dist::AbstractFloat)
```