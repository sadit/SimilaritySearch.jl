```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# Hamming distance functions

The hamming distance for binary and string data is implemented. Please recall that we use the [`evaluate`](@ref) function definition from `Distances.jl`.

## Binary hamming distance function
```@docs

BinaryHammingDistance
evaluate(::BinaryHammingDistance, a, b)
```

## Hamming distance function
```@docs

StringHammingDistance
evaluate(::StringHammingDistance, a, b)

```