```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# String distances

The following string distances are supported. Please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.


## Levenshtein (edit) distance function
```@docs

LevenshteinDistance
evaluate(::LevenshteinDistance, a, b)
```

## Longest common subsequence (LCS) distance function
```@docs

LcsDistance
evaluate(::LcsDistance, a, b)
```

## Generic Levenshtein distance function
```@docs

GenericLevenshteinDistance
evaluate(::GenericLevenshteinDistance, a, b)
```

## Common prefix dissimilarity function
```@docs

CommonPrefixDissimilarity
evaluate(::CommonPrefixDissimilarity, a, b)
```