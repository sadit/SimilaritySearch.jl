```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# String distances

The following string distances are supported. Please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.

```@docs

LevenshteinDistance
LcsDistance
GenericLevenshteinDistance
CommonPrefixDissimilarity
```