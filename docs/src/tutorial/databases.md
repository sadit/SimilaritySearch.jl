```@meta
CurrentModule = SimilaritySearch
```

# Databases: Data Abstraction with `AbstractDatabase`

Every search index in `SimilaritySearch.jl` (such as `SearchGraph` and `ExhaustiveSearch`) is generic over an [`AbstractDatabase`](@ref) rather than assuming a concrete `Matrix` or `Vector`. This page explains the architectural motivation for this abstraction and provides guidelines for selecting the appropriate database container.

---

## Motivation: Beyond Dense Fixed-Dimensional Matrices

In standard vector search benchmarks, datasets are often represented as dense matrices $M \in \mathbb{R}^{d \times n}$, where each column $i$ corresponds to an object $x_i \in \mathbb{R}^d$.

However, metric similarity search generalizes to any domain where distances between pairs of objects can be computed. In many practical applications, objects do not fit a fixed-dimensional matrix structure:
- **Variable-length collections**: Sets of prime factors, document token sequences, or transaction histories have variable cardinalities (e.g., one integer has 3 prime factors while another has 1).
- **Non-numeric structures**: Strings, categorical profiles, graphs, or custom domain structs.
- **Sparse and specialized layouts**: Compressed sparse column representations, memory-mapped files, or quantized blocks.

To accommodate these diverse representations under a unified search API, `SimilaritySearch.jl` defines the [`AbstractDatabase`](@ref) interface. Any collection that implements `Base.length`, `Base.getindex(db, i)`, and iteration can be indexed directly.

---

## `VectorDatabase`: General Object Collections

[`VectorDatabase`](@ref) is an `AbstractDatabase` container wrapping a standard Julia `Vector{T}`. It can store objects of arbitrary types and variable lengths:

```julia
julia> db = VectorDatabase([Int32[2, 3, 5], Int32[7], Int32[2, 2, 3]]);

julia> length(db), db[1], db[3]
(3, Int32[2, 3, 5], Int32[2, 2, 3])
```

Because `VectorDatabase` does not constrain `T`, it works seamlessly with arbitrary Julia data types, including strings and custom structs:

```julia
julia> words = VectorDatabase(["kitten", "sitting", "mitten", "bitten"]);

julia> words[2]
"sitting"
```

---

## `MatrixDatabase`: Zero-Copy Dense and Sparse Matrix Wrapping

When the dataset consists of fixed-dimensional numeric vectors, [`MatrixDatabase`](@ref) wraps an underlying matrix **without copying data**:

```julia
db = MatrixDatabase(X)  # db.matrix === X
```

Crucially, `MatrixDatabase` accepts any `AbstractMatrix`, including sparse matrices (`SparseMatrixCSC`) and views:

```julia
julia> using SparseArrays

julia> X = sprand(Float32, 200, 500, 0.05);  # A sparse matrix with 200 dimensions and 500 objects

julia> db = MatrixDatabase(X);

julia> db[1] isa AbstractVector
true
```

Because index algorithms interact exclusively through the `AbstractDatabase` interface, algorithms such as `SearchGraph`, `ExhaustiveSearch`, `fft`, and `allknn` execute on sparse matrices, dense matrices, or custom array backends without requiring specialized implementations.

---

## Overview of Database Types

`SimilaritySearch.jl` provides several specialized database structures suited for different storage and scalability requirements:

1. **[`MatrixDatabase`](@ref)**
   - **Characteristics**: Read-only, zero-copy wrapper for any `AbstractMatrix`.
   - **Use case**: Static, fixed-dimensional numeric datasets (dense or sparse).
2. **[`BlockMatrixDatabase`](@ref)**
   - **Characteristics**: Growable matrix storage structured as a sequence of fixed-size contiguous column blocks.
   - **Use case**: Fixed-dimensional numeric vectors that require incremental insertions (`push_item!`, `append_items!`), avoiding large reallocation costs.
3. **[`MMapMatrixDatabase`](@ref)**
   - **Characteristics**: Disk-backed, memory-mapped column storage.
   - **Use case**: Large-scale datasets that exceed available RAM, or datasets that must persist across processes. Note that updates require an explicit `flush` call for durability.
4. **[`VectorDatabase`](@ref)**
   - **Characteristics**: Flexible container for arbitrary Julia types `Vector{T}`.
   - **Use case**: Non-matrix data (sets, sequences, strings, custom structs) or variable-length representations.
5. **[`SubDatabase`](@ref)**
   - **Characteristics**: Zero-copy view mapping an index collection to a subset of another database.
   - **Use case**: Sampling, partitioning, or subset operations (produced by `view(db, indices)`, `db[indices]`, or `rand(db, k)`).

```julia
julia> sample = rand(words, 2);  # Creates a SubDatabase referencing 2 elements of `words` without copying

julia> sample isa SubDatabase
true
```

---

## Summary and Selection Guide

The following table summarizes when to choose each database container:

| Data Type & Properties | Recommended Container |
| :--- | :--- |
| Fixed-size vectors, static memory | [`MatrixDatabase`](@ref) |
| Fixed-size vectors, dynamically growing | [`BlockMatrixDatabase`](@ref) |
| Fixed-size vectors, out-of-core / disk-backed | [`MMapMatrixDatabase`](@ref) |
| Variable-length vectors, sets, strings, custom types | [`VectorDatabase`](@ref) |
| View or subset of an existing database | [`SubDatabase`](@ref) |

---

In the next section, [Distance Functions and Metric Spaces](distances.md), we examine the distance functions available in `SimilaritySearch.jl` and analyze the theoretical requirements for graph-based search.
