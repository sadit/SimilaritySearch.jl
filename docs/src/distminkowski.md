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
L2Distance
SqL2Distance
LInftyDistance
LpDistance
```