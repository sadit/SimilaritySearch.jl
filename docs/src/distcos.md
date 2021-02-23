```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# Cosine distance functions

SimilaritySearch implements some cosine/angle distance functions. Please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.
```@docs

CosineDistance
AngleDistance
NormalizedCosineDistance
NormalizedAngleDistance
```