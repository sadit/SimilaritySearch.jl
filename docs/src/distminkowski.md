```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# Minkowski family of distance functions

The following distinguished members of the Minkowski family of distance functions are provided; please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.
```@docs

L1Distance
evaluate(::L1Distance, a, b)
```

## Euclidean distance
```@docs
L2Distance
evaluate(::L2Distance, a, b)
```

## Squared euclidean distance
```@docs
SqL2Distance
evaluate(::SqL2Distance, a, b)
```

## Chebyshev distance
```@docs
LInftyDistance
evaluate(::LInftyDistance, a, b)

```

## Generic Minkowski distance functions
```@docs
LpDistance
evaluate(::LpDistance, a, b)

```