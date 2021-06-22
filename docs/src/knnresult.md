```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```

# KnnResult

SimilaritySearch's core is to solve knn searches; for this matter, it relies on a fixed size priority queue, the `KnnResult` struct and its related functions.
The role of this struct is specifying and handling results.

Its usage without searches is pretty simple, as it is exemplified in the next code.
```@example
using SimilaritySearch

res = KnnResult(3)
for i in 1:3
    push!(res, i => rand())
end

println("first 3: ", collect(res))

for i in 4:10
    push!(res, i => rand())
end

println("final: ", collect(res))
```

## Data structures
The `KnnResult` contains two aligned arrays 'id' and 'dist' and an indicator of its maximum capacity 'k'.
```@docs
KnnResult
```

## Length and capacity of the queue
```@example
using SimilaritySearch

res = KnnResult(3)
for i in 1:5
    push!(res, i => rand())
    println("current length: $(length(res)), capacity: $(maxlength(res)), first-dist: $(first(res).dist), last-dist: $(last(res).dist), covrad: $(covrad(res))")
end

```

The priority queue has a current length and a maximum capacity

```@docs
length(::KnnResult)
maxlength
```

### Bounds of the queue
```@docs
covrad
last(::KnnResult)
first(::KnnResult)
```

## Accessing and iterating
```@docs
getindex(::KnnResult, i)
eachindex(::KnnResult)
iterate(::KnnResult)
```

## Adding and removing items
```@docs
empty!
push!(::KnnResult, id, dist)
popfirst!(::KnnResult)
pop!(::KnnResult)
```