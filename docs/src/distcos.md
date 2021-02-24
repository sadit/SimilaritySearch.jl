```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# Cosine distance functions

SimilaritySearch implements some cosine/angle distance functions. Please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.

## Cosine distance function
```@docs
CosineDistance
evaluate(::CosineDistance, a, b)
```

## Angle distance function
```@docs
AngleDistance
evaluate(::AngleDistance, a, b)
```

## Normalized cosine distance function
```@docs
NormalizedCosineDistance
evaluate(::NormalizedCosineDistance, a, b)

```

## Normalized angle distance function
```@docs
NormalizedAngleDistance
evaluate(::NormalizedAngleDistance, a, b)
```