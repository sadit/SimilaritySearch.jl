```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# Set distances

This functions use sets measure the distance between them; each set is represented as an array of sorted integers. 
Please recall that we use the [`evaluate`](@ref) function definition from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl).
The following set distances are supported.

## Jaccard distance function

```@docs
JaccardDistance
evaluate(::JaccardDistance, a, b)
```

## Dice distance function
```@docs
DiceDistance
evaluate(::DiceDistance, a, b)
```

## Intersection dissimilarity function
```@docs
IntersectionDissimilarity
evaluate(::IntersectionDissimilarity, a, b)
```