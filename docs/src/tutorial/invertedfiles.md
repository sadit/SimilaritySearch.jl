```@meta
CurrentModule = SimilaritySearch
```

# Inverted Files and Posting List Intersections

An inverted index maps distinct feature components (such as vocabulary terms, set elements, or non-zero coordinate dimensions) to **posting lists** containing the identifiers of documents containing those components. Inverted files provide exact search capabilities for high-dimensional sparse vectors, set metrics, and Maximum Inner Product Search (MIPS).

`SimilaritySearch.jl` implements inverted indexes in the `InvertedFiles` module (`SimilaritySearch.InvertedFiles`) and list intersection routines in the `Intersections` module (`SimilaritySearch.Intersections`).

---

## Architectural Models of `InvertedFiles`

`SimilaritySearch.InvertedFiles` provides two primary storage backends for posting lists:

1. **Array Backend ([`InvertedFiles.InvertedFile`](@ref))**:
   - Stores posting lists in a contiguous array of type `AdjList(UInt32)`.
   - Keys are represented as dense integer indices $k \in \{1, \dots, \text{vocsize}\}$.
   - Provides high cache locality and minimal access overhead when the vocabulary is known and bounded.
2. **Dictionary Backend ([`InvertedFiles.DictInvertedFile`](@ref))**:
   - Stores posting lists in a hash map of type `AdjDict{KeyType, UInt32}`.
   - Keys can be arbitrary Julia types (such as `String`, `NTuple`, or sparse non-contiguous integer IDs).
   - Dynamically allocates posting lists only for tokens observed in the indexed documents, avoiding pre-allocation overhead in open or high-cardinality universes.

### Candidate Evaluation and Distance Computation

The search procedure processes candidates according to the specified distance metric:

- **Exact Fast-Path Set Metrics**: For standard set metrics ([`Dist.Sets.Jaccard`](@ref Dist.Sets.Jaccard), [`Dist.Sets.Dice`](@ref Dist.Sets.Dice), [`Dist.Sets.Intersection`](@ref Dist.Sets.Intersection), [`Dist.Sets.CosineSet`](@ref Dist.Sets.CosineSet), and [`Dist.Sets.RogersTanimoto`](@ref Dist.Sets.RogersTanimoto)), the metric distance is computed analytically during the posting list intersection pass.
- **Direct Candidate Evaluation**: For other metrics (such as [`Dist.NormCosine`](@ref Dist.NormCosine) for sparse vectors or arbitrary user metrics), posting lists identify candidate documents sharing non-zero coordinates, and `search` evaluates the metric directly against the candidate objects stored in `database(idx)`.

---

## Worked Example: Recipe Ingredient Indexing

We illustrate inverted indexing using a dataset of recipe ingredient collections, demonstrating set queries, metric rankings, and weighted sparse vector search.

```julia
using SimilaritySearch, SimilaritySearch.InvertedFiles

ingredients = ["flour", "sugar", "egg", "butter", "milk", "salt",
               "yeast", "tomato", "cheese", "basil", "chicken", "rice"]
id(name) = findfirst(==(name), ingredients)
vocsize = length(ingredients)   # 12

recipe_sets = Dict(
    "Pancakes"        => UInt32[id("flour"), id("sugar"), id("egg"), id("milk")],
    "Bread"           => UInt32[id("flour"), id("salt"), id("yeast")],
    "MargheritaPizza" => UInt32[id("flour"), id("tomato"), id("cheese"), id("basil")],
    "Omelette"        => UInt32[id("egg"), id("butter"), id("milk"), id("cheese")],
    "ChickenRice"     => UInt32[id("chicken"), id("rice"), id("salt")],
    "Cheesecake"      => UInt32[id("sugar"), id("egg"), id("butter"), id("cheese")],
)
names = collect(keys(recipe_sets))
sets = VectorDatabase([sort!(recipe_sets[n]) for n in names])

ctx = InvertedFileContext()
q = UInt32[id("flour"), id("cheese")]   # Query: items containing flour and/or cheese
```

### Conjunction (AND) Query

The parameter `t` defines the minimum number of matching query tokens required for candidate retention. Setting $t = |Q|$ enforces a strict set intersection (logical AND):

```julia
IJ = InvertedFile(vocsize, Dist.Sets.Jaccard())
append_items!(IJ, ctx, sets)

res = knnqueue(ctx, 6)
search(IJ, ctx, q, res; t=length(q))  # t = 2 requires matches for both 'flour' and 'cheese'

for it in IdDistView(res)
    println(names[it.id], " => ", it.dist)
end
# Output: MargheritaPizza => 0.5
```

### Disjunction (OR) Query and Jaccard Ranking

Setting $t = 1$ retrieves all documents containing at least one query component (logical OR), ranking candidates by Jaccard distance:

```julia
res = knnqueue(ctx, 6)
search(IJ, ctx, q, res; t=1)

for it in IdDistView(res)
    println(names[it.id], " => ", it.dist)
end
```

### Rogers-Tanimoto Ranking

[`Dist.Sets.RogersTanimoto`](@ref Dist.Sets.RogersTanimoto) incorporates mutual absences relative to the total vocabulary size $\sigma$:

```julia
IR = InvertedFile(vocsize, Dist.Sets.RogersTanimoto(vocsize))
append_items!(IR, ctx, sets)

res = knnqueue(ctx, 6)
search(IR, ctx, q, res; t=1)

for it in IdDistView(res)
    println(names[it.id], " => ", it.dist)
end
```

---

## Weighted Sparse Vector Search: `NormCosine`

To perform similarity search on weighted attributes (e.g., ingredient quantities or TF-IDF weights), represent objects as unit-normalized sparse vectors:

```julia
using SparseArrays, LinearAlgebra

l2normalize(idx::Vector{<:Integer}, val::Vector{<:Real}, n) = normalize!(sparsevec(idx, Float32.(val), n))

recipe_weights = Dict(
    "Pancakes"        => l2normalize([id("flour"), id("sugar"), id("egg"), id("milk")], [3.0, 1.0, 2.0, 2.0], vocsize),
    "Bread"           => l2normalize([id("flour"), id("salt"), id("yeast")], [4.0, 1.0, 1.0], vocsize),
    "MargheritaPizza" => l2normalize([id("flour"), id("tomato"), id("cheese"), id("basil")], [3.0, 2.0, 2.0, 1.0], vocsize),
    "Omelette"        => l2normalize([id("egg"), id("butter"), id("milk"), id("cheese")], [3.0, 1.0, 1.0, 2.0], vocsize),
    "ChickenRice"     => l2normalize([id("chicken"), id("rice"), id("salt")], [2.0, 2.0, 1.0], vocsize),
    "Cheesecake"      => l2normalize([id("sugar"), id("egg"), id("butter"), id("cheese")], [2.0, 2.0, 1.0, 3.0], vocsize),
)
weights = VectorDatabase([recipe_weights[n] for n in names])

W = InvertedFile(vocsize, Dist.NormCosine())
append_items!(W, ctx, weights)

qw = l2normalize([id("flour"), id("egg"), id("cheese")], [1.0, 1.0, 1.0], vocsize)
res = knnqueue(ctx, 6)
search(W, ctx, qw, res)

for it in IdDistView(res)
    println(names[it.id], " => ", it.dist)
end
```

---

## Arbitrary Key Indexing with `DictInvertedFile`

To index native types (such as `String`, tuples, or open vocabulary terms) directly without mapping them to a contiguous integer range, use [`DictInvertedFile`](@ref):

```julia
recipes = Dict(
    "Pancakes"        => Set(["flour", "sugar", "egg", "milk"]),
    "Bread"           => Set(["flour", "salt", "yeast"]),
    "MargheritaPizza" => Set(["flour", "tomato", "cheese", "basil"]),
    "Omelette"        => Set(["egg", "butter", "milk", "cheese"]),
    "ChickenRice"     => Set(["chicken", "rice", "salt"]),
    "Cheesecake"      => Set(["sugar", "egg", "butter", "cheese"]),
)

names = collect(keys(recipes))
db_strings = VectorDatabase([recipes[n] for n in names])

# Instantiate a dictionary-backed inverted file
IDict = DictInvertedFile(String, Dist.Sets.Jaccard())
ctx_dict = getcontext(IDict)
append_items!(IDict, ctx_dict, db_strings)

q_str = Set(["flour", "cheese"])
res = knnqueue(ctx_dict, 6)
search(IDict, ctx_dict, q_str, res; t=1)

for it in IdDistView(res)
    println(names[it.id], " => ", it.dist)
end
```

---

## Extending `InvertedFile` via `identiterator`

When constructing posting lists, `InvertedFile` extracts feature identifiers from objects using [`InvertedFiles.identiterator`](@ref):

```julia
identiterator(dist::PreMetric, obj)
```

By default, `identiterator` supports `Set`, sorted integer vectors, `Dict`, and `SparseVector`. To index custom object representations or extract specialized tokens for a particular distance function, define a method overload for `identiterator(dist::MyDistance, obj::MyType)`.

---

## Posting List Intersection Algorithms (`Intersections`)

The `Intersections` module provides low-level algorithms for multi-list intersection and union operations:

- `Intersections.svs`: Smallest-Vector-First intersection.
- `Intersections.bk!` / `Intersections.bkt!`: Adaptive threshold intersection (Baeza-Yates / WAND style).
- `Intersections.umerge!` / `Intersections.imerge!`: Union and intersection algorithms across posting lists.
- `Intersections.xmerge!`: Thresholded multi-way list merge ($t$-occurrence filter).

```julia
using SimilaritySearch.Intersections

list1 = [1, 3, 5, 7, 9]
list2 = [1, 2, 3, 4, 5]

common = svs([list1, list2])  # Returns [1, 3, 5]
```

---

In the next section, [Quantization and Bit Sketches](quantization_and_bitsketches.md), we explore vector compression and binary projection methods for high-throughput similarity search.
