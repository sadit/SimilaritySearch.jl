```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# Set distances

The following set distances are supported. Please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.

```@docs

JaccardDistance
DiceDistance
IntersectionDissimilarity
```