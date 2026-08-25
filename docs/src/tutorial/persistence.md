```@meta
CurrentModule = SimilaritySearch
```

# Index Persistence and Serialization

Constructing a [`SearchGraph`](@ref) requires evaluating nearest neighbors for each inserted object. To avoid repeating this computation across application restarts, search indexes can be serialized to disk and reloaded using [JLD2.jl](https://github.com/JuliaIO/JLD2.jl).

Because all indexes in `SimilaritySearch.jl` are standard Julia structs, they can be directly serialized without specialized conversion layers.

```julia
] add JLD2 Accessors
```

---

## Basic Serialization and Deserialization

To save an index to disk, use `jldsave`. To restore it, use `load_object`:

```julia
using SimilaritySearch, JLD2

# 1. Prepare data and construct index
function primes_upto(n::Integer)
    sieve = trues(n)
    sieve[1] = false
    for p in 2:isqrt(n)
        sieve[p] && (sieve[p*p:p:n] .= false)
    end
    findall(sieve)
end

function prime_gap_windows(n::Integer, w::Integer)
    P = primes_upto(n)
    gaps = Float32.(log2.(diff(P)))
    m = length(gaps) - w
    M = Matrix{Float32}(undef, w, m)
    for i in 1:m
        M[:, i] .= view(gaps, i:i+w-1)
    end
    M
end

X = MatrixDatabase(prime_gap_windows(200_000, 5))
G = SearchGraph(Dist.SqL2(), X)
ctx = SearchGraphContext()
index!(G, ctx)

# 2. Serialize index to disk
jldsave("graph.jld2"; G)

# 3. Deserialize index in a subsequent session
G2 = load_object("graph.jld2")

# 4. Verify identical search behavior
res1 = knnqueue(ctx, 5); search(G, ctx, X[1], res1)
res2 = knnqueue(ctx, 5); search(G2, ctx, X[1], res2)
collect(IdView(res1)) == collect(IdView(res2))   # Returns true
```

The same serialization procedure applies to [`ExhaustiveSearch`](@ref) and [`SimilaritySearch.Exact.ParallelExhaustiveSearch`](@ref).

---

## Persisting Indexes vs. Contexts

Only the index structure (`G`) should be persisted. 

Search contexts ([`SearchGraphContext`](@ref) and [`GenericContext`](@ref)) contain transient execution state, including thread scratch buffers (`vstates`, `beams`) and logger references. Always instantiate a fresh context (`ctx = SearchGraphContext()`) after deserializing an index.

---

## Decoupling Graph Topology from Dataset Storage

When working with large datasets, the raw vectors may already be stored in an external system (such as an HDF5 repository, a memory-mapped file, or a database). Serializing a duplicate copy of the dataset inside the index file causes unnecessary disk and memory usage.

To persist only the graph topology:
1. Substitute the database field with an empty placeholder database before saving.
2. Re-attach the actual dataset after loading.

This can be accomplished using `Accessors.@reset`:

```julia
using Accessors

# 1. Substitute an empty placeholder database
placeholder = MatrixDatabase(zeros(Float32, 2, 2))
G_topology = @reset G.db = placeholder

# 2. Save only the graph topology
jldsave("graph_topology.jld2"; G_topology)

# 3. Load topology and reattach the primary dataset
loaded = load_object("graph_topology.jld2")
X_real = MatrixDatabase(prime_gap_windows(200_000, 5))  # Or loaded from an external HDF5 / mmap file
loaded = @reset loaded.db = X_real
```

The adjacency graph references numeric object identifiers in the range $1, \dots, |X|$. Reattaching the original dataset restores search functionality immediately without data duplication.

---

## Schema Evolution and Versioning

JLD2 serializes Julia struct type definitions. If the internal schema of an index type changes between major package versions, deserialization may require explicit migration or re-indexing. For production deployments across versions, verify compatibility or maintain reproducible package environments (`Manifest.toml`).

---

In the next section, [Logging and Observation Channels](logging.md), we examine the dual-channel architecture for progress monitoring and incremental graph observation.
