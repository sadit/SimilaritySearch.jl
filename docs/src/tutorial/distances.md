```@meta
CurrentModule = SimilaritySearch
```

# A gallery of distances

`SimilaritySearch.jl` doesn't hard-code "a vector of `Float32`" as *the* notion of an
object -- any type that a distance function's `evaluate` method accepts can be indexed.
Distance functions live under `Dist` and its submodules:

| Submodule    | For objects that are...                              | A few examples |
|--------------|-------------------------------------------------------|----------------|
| `Dist`       | numeric vectors                                        | [`L1`](@ref Dist.L1), [`L2`](@ref Dist.L2), [`SqL2`](@ref Dist.SqL2), [`LInfty`](@ref Dist.LInfty), [`Lp`](@ref Dist.Lp), [`Cosine`](@ref Dist.Cosine), [`Angle`](@ref Dist.Angle) |
| `Dist.Sets`  | sets, given as **sorted** vectors of comparable items   | `Jaccard`, `Dice`, `Intersection`, `CosineSet`, `RogersTanimoto` |
| `Dist.Seqs`  | sequences (order matters, any element type)             | `Levenshtein`, `LCS`, `CommonPrefix`, `Hamming` |
| `Dist.Bits`  | bit strings, given as arrays of `Unsigned`/`Bool`s      | `Hamming`, `RogersTanimoto` |

All examples below use [`ExhaustiveSearch`](@ref) -- exact, and instant at this scale --
so we can focus entirely on what each distance actually measures. The **last section**
explains why several of these (all the `Dist.Sets`/`Dist.Seqs`/`Dist.Bits` ones) are a
poor match for [`SearchGraph`](@ref) specifically, regardless of how well they work for
`ExhaustiveSearch`.

## Vectors: `L1`/`L2`/`SqL2`/`LInfty`/`Lp`, and why `Cosine` is different in kind

These all take plain numeric vectors. The `Lp` family only differs in *how* per-coordinate
differences are combined:

```julia
using SimilaritySearch, Distances

u = Float32[0, 0]
v = Float32[3, 4]

evaluate(Dist.L2(), u, v)      # 5.0  == sqrt(3^2 + 4^2)
evaluate(Dist.SqL2(), u, v)    # 25.0 == 3^2 + 4^2 (skips the sqrt -- cheaper, same ranking as L2)
evaluate(Dist.L1(), u, v)      # 7.0  == |3| + |4|
evaluate(Dist.LInfty(), u, v)  # 4.0  == max(|3|, |4|)
evaluate(Dist.Lp(3.0), u, v)   # (3^3 + 4^3)^(1/3) ≈ 4.498
```

`Cosine`/`Angle`, in contrast, ignore vector *magnitude* entirely and only compare
*direction* -- a different notion of "similar" than any `Lp` distance, not just a
different formula:

```julia
u  = Float32[1, 0]
v  = Float32[2.5, 0]  # same direction as u, much larger magnitude
w  = Float32[-1, 0]   # opposite direction, same magnitude as u

evaluate(Dist.L2(), u, v)      # 1.5   -- far apart by L2 (different magnitude)
evaluate(Dist.Cosine(), u, v)  # 0.0   -- identical by Cosine (same direction)

evaluate(Dist.L2(), u, w)      # 2.0
evaluate(Dist.Cosine(), u, w)  # 2.0   -- opposite directions are as far apart as Cosine gets
```

Whether that's what you want depends entirely on your application: if only the
*direction* of a vector is meaningful (as is common for normalized embeddings or
term-weight vectors), use `Cosine`/`Angle`; if magnitude matters, use one of the `Lp`
family instead. `NormCosine`/`NormAngle` are cheaper variants for when inputs are
already known to be unit vectors.

## Sets: comparing numbers by their prime factors

Represent each integer by the *set* of its distinct prime factors, given as a sorted
vector (all `Dist.Sets` distances expect sorted inputs, so they can compute
intersections/unions with a linear merge instead of a hash set):

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

factors(60)   # Int32[2, 3, 5]   (60 = 2²·3·5)
factors(90)   # Int32[2, 3, 5]   (90 = 2·3²·5) -- a *different* number, the *same* factor set!
factors(97)   # Int32[97]        (97 is prime)
```

```julia
a, b, c = factors(60), factors(90), factors(97)

evaluate(Dist.Sets.Jaccard(), a, b)      # 0.0 -- identical sets: {2,3,5} vs {2,3,5}
evaluate(Dist.Sets.Dice(), a, b)         # 0.0
evaluate(Dist.Sets.Intersection(), a, b) # 0.0
evaluate(Dist.Sets.CosineSet(), a, b)    # ≈ 0.0

evaluate(Dist.Sets.Jaccard(), a, c)      # 1.0 -- {2,3,5} and {97} share nothing
```

That `0.0` between 60 and 90 is not a bug: `Jaccard`/`Dice`/`Intersection`/`CosineSet`
only see *which* primes divide a number, never how many times, nor the number itself --
so any two numbers with the same distinct prime factors tie exactly. This is a first
glimpse of something we'll come back to: set distances built this way take on relatively
few distinct values and produce lots of exact ties. [`RogersTanimoto`](@ref
Dist.Sets.RogersTanimoto) additionally needs the universe size (here, the count of
primes up to your largest `n`) to account for the primes *neither* number is divisible
by.

Indexing and querying this space:

```julia
n = 1000
X = VectorDatabase([factors(i) for i in 1:n])
idx = ExhaustiveSearch(Dist.Sets.Dice(), X)
ctx = GenericContext()
res = knnqueue(ctx, 5)
search(idx, ctx, factors(360), res)   # numbers that share 360's prime factors {2,3,5}
```

## Sequences: the *same* numbers, now order- and multiplicity-sensitive

Instead of the *set* of prime factors, use the full factorization *sequence*, repeats
and all (`360 = 2·2·2·3·3·5`, not just `{2,3,5}`). This is exactly the same underlying
information the sets above discarded -- which is the point: same integers, different
representation, different notion of "close":

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

evaluate(Dist.Seqs.Levenshtein(), a, b)   # 1.0 -- one substitution turns [2,2,3,5] into [2,3,3,5]
evaluate(Dist.Seqs.LCS(), a, b)           # 2.0 -- edit distance allowing only insert/delete (no substitution)
evaluate(Dist.Seqs.CommonPrefix(), a, b)  # 0.75 -- only the leading "2" matches before they diverge
```

Contrast this with the previous section: 60 and 90 were *identical* by `Jaccard` (same
factor set) but *distance 1* by `Levenshtein` (different factor sequence) -- neither
answer is "more correct," they're answering different questions about the same numbers.
`Dist.Seqs.Hamming` is also available here, but (unlike `Levenshtein`/`LCS`) it requires
both sequences to already be the same length, since it compares position-by-position with
no notion of insertion/deletion.

## Bit patterns: divisibility fingerprints

One more lens on the same integers: fix a list of small primes and record, as a single
bit string, which of them divide `n`. Two numbers that happen to share the same small
prime factors will produce the *exact same bit pattern*, even though (as with the sets
above) they're different numbers:

```julia
function primes_upto(n::Integer)
    sieve = trues(n)
    sieve[1] = false
    for p in 2:isqrt(n)
        sieve[p] && (sieve[p*p:p:n] .= false)
    end
    findall(sieve)
end

smallprimes = primes_upto(400)[1:64]   # first 64 primes -- one bit each

function signature(n::Integer, ps::Vector{Int})
    s = zero(UInt64)
    for (i, p) in enumerate(ps)
        n % p == 0 && (s |= UInt64(1) << (i - 1))
    end
    s
end

s60, s90, s97 = signature(60, smallprimes), signature(90, smallprimes), signature(97, smallprimes)

evaluate(Dist.Bits.Hamming(), s60, s90)  # 0.0 -- 60 and 90 share the same small-prime divisors {2,3,5}
evaluate(Dist.Bits.Hamming(), s60, s97)  # 4.0 -- 97 differs in 4 of the tracked bits
```

Once again 60 and 90 tie exactly at distance `0`. With only 64 possible bit positions,
and most numbers sharing small factors like 2 and 3 with many others, exact ties are the
rule here, not the exception -- there just isn't much "resolution" to rank numbers
finely apart. Keep this in mind; it's exactly the property the next section is about.

## Why `SearchGraph` should not be used with discrete/combinatorial distances

**Rule of thumb: if a distance's possible output values form a small, discrete set with
lots of ties (as with all three `Dist.Sets`/`Dist.Seqs`/`Dist.Bits` examples above), use
[`ExhaustiveSearch`](@ref), not [`SearchGraph`](@ref), regardless of dataset size.**

`SearchGraph`'s `BeamSearch` is a *greedy local search* over a proximity graph: starting
from some entry point, it repeatedly hops to a neighbor that's closer to the query than
the current best candidate, until no such neighbor exists. This only reliably converges
to the true nearest neighbors if the space is **navigable**: from (almost) anywhere,
there's a neighbor that's strictly closer to the query, and being closer in that
step-by-step sense actually correlates with being closer overall. Continuous vector
distances (`L2`, `Cosine`, ...) generally have this property -- there's essentially
always *some* direction of improvement, because the distance takes on a huge range of
distinct real values.

Discrete/combinatorial distances routinely break this assumption: with few distinct
output values and many exact ties (like `Jaccard`/`Dice`/`Hamming` above), the graph's
greedy walk can easily reach a candidate where *no* neighbor is *strictly* closer, even
though better candidates exist elsewhere in the graph -- the search simply has nothing
to climb down onto next and stops early. The practical symptom is degraded recall that
doesn't reliably improve by tuning `BeamSearch`'s parameters (`bsize`, hints, ...) the
way it would for a continuous space, because the underlying problem isn't
under-exploration, it's a lack of gradient to follow in the first place. `ExhaustiveSearch`
has no such issue since it doesn't rely on navigability at all -- it always checks every
element -- and it is fast enough for exactly this kind of small-alphabet, symbolic data
in practice.

If you need approximate (sub-linear) search over a genuinely discrete/combinatorial
space at large scale, that's a real research problem this package doesn't attempt to
solve for you -- techniques like locality-sensitive hashing or inverted indices tailored
to the specific distance are the usual answer, not a proximity graph.

## Further reading

For very large numeric datasets where memory (not navigability) is the bottleneck, see
the `ScalarQuant` (`ScalarQuant.SQu8`, `.SQu4`, ...) and `Projections`
(`SimilaritySearch.Projections`) submodules in the [API reference](../api.md) -- they trade
some accuracy for a much smaller memory footprint per vector, and combine with either
`ExhaustiveSearch` or `SearchGraph`. They're outside the scope of this tutorial, but
worth knowing they exist.

Next: [`SearchGraph`, in depth](searchgraph.md), using one of the continuous, genuinely
navigable spaces this section built (prime *gaps*, not prime factors).
