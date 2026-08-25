```@meta
CurrentModule = SimilaritySearch
```

# Distance Functions and Metric Spaces

In `SimilaritySearch.jl`, search indexes are decoupled from specific object representations. Any data type supported by a distance function's `evaluate` method can be indexed. 

Distance functions are organized within the `Dist` module and its specialized submodules:

| Submodule | Target Object Domain | Representative Distance Functions |
| :--- | :--- | :--- |
| `Dist` | Real-valued dense or sparse vectors ($\mathbb{R}^d$) | [`L1`](@ref Dist.L1), [`L2`](@ref Dist.L2), [`SqL2`](@ref Dist.SqL2), [`LInfty`](@ref Dist.LInfty), [`Lp`](@ref Dist.Lp), [`Cosine`](@ref Dist.Cosine), [`Angle`](@ref Dist.Angle) |
| `Dist.Sets` | Sets, represented as **sorted** vectors of distinct comparable elements | `Jaccard`, `Dice`, `Intersection`, `CosineSet`, `RogersTanimoto` |
| `Dist.Seqs` | Ordered sequences (strings, token arrays) | `Levenshtein`, `LCS`, `CommonPrefix`, `Hamming` |
| `Dist.Bits` | Binary vectors and bit strings (`Unsigned`, `BitVector`) | `Hamming`, `RogersTanimoto` |

All examples in this section use [`ExhaustiveSearch`](@ref) to illustrate the properties of each distance metric. The final section provides a theoretical analysis of why graph-based indexes ([`SearchGraph`](@ref)) require continuous distance spaces and should not be used with discrete or combinatorial metrics.

---

## Vector Distances: $L_p$ Norms and Angular Metrics

### $L_p$ Metrics

The $L_p$ family computes distance based on coordinate-wise absolute differences:

$$d_{L_p}(u, v) = \left( \sum_{i=1}^d |u_i - v_i|^p \right)^{1/p}$$

```julia
using SimilaritySearch, Distances

u = Float32[0, 0]
v = Float32[3, 4]

evaluate(Dist.L2(), u, v)      # 5.0  = sqrt(3² + 4²) (Euclidean distance)
evaluate(Dist.SqL2(), u, v)    # 25.0 = 3² + 4² (Squared Euclidean, avoids square root)
evaluate(Dist.L1(), u, v)      # 7.0  = |3| + |4| (Manhattan distance)
evaluate(Dist.LInfty(), u, v)  # 4.0  = max(|3|, |4|) (Chebyshev distance)
evaluate(Dist.Lp(3.0), u, v)   # (3³ + 4³)^(1/3) ≈ 4.498
```

`Dist.SqL2` is strictly monotonic with respect to `Dist.L2`. It preserves nearest-neighbor rankings while eliminating the computational overhead of the square root operation.

### Directional and Angular Distances

Unlike $L_p$ metrics, `Cosine` and `Angle` measure directional alignment, ignoring vector magnitude:

$$d_{\text{Cosine}}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

```julia
u = Float32[1, 0]
v = Float32[2.5, 0]  # Same direction as u, larger magnitude
w = Float32[-1, 0]   # Opposite direction, same magnitude

evaluate(Dist.L2(), u, v)      # 1.5 -- distinct due to magnitude differences
evaluate(Dist.Cosine(), u, v)  # 0.0 -- identical direction

evaluate(Dist.L2(), u, w)      # 2.0
evaluate(Dist.Cosine(), u, w)  # 2.0 -- diametrically opposed directions
```

When vectors are normalized such that $\|u\|_2 = 1$, [`Dist.NormCosine`](@ref Dist.NormCosine) and [`Dist.NormAngle`](@ref Dist.NormAngle) provide optimized evaluations by omitting the denominator normalization.

---

## Set Distances: Prime Factor Representations

Consider an integer $n \in \mathbb{N}$ represented by the set of its distinct prime factors, formatted as a sorted `Vector{Int32}`:

```julia
function factors(n::Integer)
    f = Int32[]
    m = n
    d = Int32(2)
    while d * d <= m
        if m % d == 0
            push!(f, d)
            while m % d == 0
                m ÷= d
            end
        end
        d += 1
    end
    m > 1 && push!(f, m)
    isempty(f) ? Int32[1] : f
end

factors(60)   # Int32[2, 3, 5] (60 = 2² · 3 · 5)
factors(90)   # Int32[2, 3, 5] (90 = 2 · 3² · 5)
factors(97)   # Int32[97]      (97 is prime)
```

Distance functions in `Dist.Sets` assume sorted inputs to compute set operations via linear merges in $O(|A| + |B|)$ time:

```julia
a, b, c = factors(60), factors(90), factors(97)

evaluate(Dist.Sets.Jaccard(), a, b)      # 0.0 -- identical prime support: {2, 3, 5} vs {2, 3, 5}
evaluate(Dist.Sets.Dice(), a, b)         # 0.0
evaluate(Dist.Sets.Intersection(), a, b) # 0.0
evaluate(Dist.Sets.CosineSet(), a, b)    # 0.0

evaluate(Dist.Sets.Jaccard(), a, c)      # 1.0 -- disjoint sets: {2, 3, 5} ∩ {97} = ∅
```

The distance between $60$ and $90$ is `0.0` because set metrics evaluate support rather than multiplicity. For [`RogersTanimoto`](@ref Dist.Sets.RogersTanimoto), the size of the underlying universe $\sigma$ (the total number of primes considered) must be specified to account for mutual non-occurrences.

Building an exact search index on this set space:

```julia
n = 1000
X = VectorDatabase([factors(i) for i in 1:n])
idx = ExhaustiveSearch(Dist.Sets.Dice(), X)
ctx = GenericContext()
res = knnqueue(ctx, 5)
search(idx, ctx, factors(360), res)   # Finds integers sharing prime factors {2, 3, 5}
```

---

## Sequence Distances: Preserving Multiplicity and Order

To capture factor multiplicity and ordering, we can represent an integer by its complete prime factorization sequence:

```julia
function factor_sequence(n::Integer)
    f = Int32[]
    m = n
    d = Int32(2)
    while d * d <= m
        while m % d == 0
            push!(f, d)
            m ÷= d
        end
        d += 1
    end
    m > 1 && push!(f, m)
    isempty(f) ? Int32[1] : f
end

a = factor_sequence(60)   # Int32[2, 2, 3, 5]
b = factor_sequence(90)   # Int32[2, 3, 3, 5]

evaluate(Dist.Seqs.Levenshtein(), a, b)   # 1.0  -- one substitution transforms a into b
evaluate(Dist.Seqs.LCS(), a, b)           # 2.0  -- Longest Common Subsequence edit distance
evaluate(Dist.Seqs.CommonPrefix(), a, b)  # 0.75 -- length normalized prefix mismatch
```

While $60$ and $90$ have identical set representations ($d_{\text{Jaccard}} = 0.0$), their factorization sequences differ ($d_{\text{Levenshtein}} = 1.0$). 

For sequences of equal length, [`Dist.Seqs.Hamming`](@ref Dist.Seqs.Hamming) evaluates coordinate-wise mismatches without allowing insertions or deletions.

---

## Bit Patterns: Binary Divisibility Fingerprints

Another representation maps each integer to a binary signature indicating divisibility by the first $k$ prime numbers:

```julia
function primes_upto(n::Integer)
    sieve = trues(n)
    sieve[1] = false
    for p in 2:isqrt(n)
        sieve[p] && (sieve[p*p:p:n] .= false)
    end
    findall(sieve)
end

smallprimes = primes_upto(400)[1:64]   # First 64 prime numbers

function signature(n::Integer, ps::Vector{Int})
    s = zero(UInt64)
    for (i, p) in enumerate(ps)
        n % p == 0 && (s |= UInt64(1) << (i - 1))
    end
    s
end

s60, s90, s97 = signature(60, smallprimes), signature(90, smallprimes), signature(97, smallprimes)

evaluate(Dist.Bits.Hamming(), s60, s90)  # 0.0 -- identical divisibility bits for small primes
evaluate(Dist.Bits.Hamming(), s60, s97)  # 4.0 -- differs in 4 bit positions
```

---

## Navigability and Graph Search: Continuous vs. Discrete Metrics

A central principle in metric similarity search is the distinction between **navigable continuous metric spaces** and **discrete combinatorial spaces**.

### The Navigability Hypothesis in Graph-Based Search

[`SearchGraph`](@ref) implements beam search over a proximity graph. Starting from one or more entry vertices (hints), the search algorithm greedily traverses the graph by moving to adjacent vertices that are strictly closer to the query $q$, terminating when no neighboring vertex improves upon the current best candidates.

This greedy exploration converges to the true nearest neighbors if the metric space is **navigable**:
1. For almost every vertex $u$, there exists a neighbor $v$ such that $d(v, q) < d(u, q)$.
2. The distance function provides a fine-grained gradient of real values that guides the beam search towards the global optimum.

Continuous vector spaces ($\mathbb{R}^d$ under $L_2$, $\text{Sq}L_2$, or Cosine) satisfy this condition because distances vary continuously across a continuum of values.

```
Continuous Space (Smooth Gradient):
Query: q
  u ──────> v ──────> w ──────> Nearest Neighbor
d: 4.8      3.2       1.1       0.2
(Greedy search follows strictly decreasing distances)
```

### The Plateau Problem in Discrete Spaces

In discrete and combinatorial spaces (such as Jaccard, edit distance on short strings, or Hamming distance on small bit arrays), the set of possible distance values is small and finite. Consequently:
- Large subsets of objects have identical distance values (equidistant plateaus).
- Greedy search easily reaches a local minimum where all immediate neighbors have $d(v, q) \ge d(u, q)$, even though closer objects exist in other regions of the graph.
- Increasing the beam size or search parameters fails to resolve the fundamental absence of a distance gradient.

```
Discrete / Combinatorial Space (Plateau / Zero Gradient):
Query: q
  u ──────> v ──────> ? (Search stalls)
d: 1.0      1.0       1.0 (All neighbors tie at distance 1.0)
```

### Practical Recommendation

- **Continuous vector spaces ($\mathbb{R}^d$, embeddings)**: Use [`SearchGraph`](@ref) for sub-linear approximate nearest neighbor search, or [`ExhaustiveSearch`](@ref) for baseline validation.
- **Discrete, combinatorial, or set spaces (Jaccard, Levenshtein, Hamming)**: Use [`ExhaustiveSearch`](@ref) or inverted indexes ([`InvertedFile`](@ref)). Do not use `SearchGraph` for discrete metrics with high tie frequencies.

---

In the next section, [`SearchGraph`, in depth](searchgraph.md), we explore the proximity graph index using continuous geometric vectors.
