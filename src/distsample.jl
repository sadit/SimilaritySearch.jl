# This file is part of SimilaritySearch.jl
export distsample_ut, distsample

"""
    distsample_ut(dist::SemiMetric, X::AbstractDatabase; prob::Float64=0.01, samplesize=0) -> S

Computes a sample of the upper triangular pairwise distance matrix.
Returns an array of distances of close to ``prob \\cdot n^2/2`` entries for a database of size ``n``.
This method is fine to work with small datasets (not million-sized datasets); this method does not
return duplicates nor symmetric duplicates.

# Arguments
- `dist`: distance function
- `X`: input database

# Keyword Arguments
- `prob`: sampling probability (on the upper triangle pairwise distance matrix)
- `samplesize`: if given (`> 0`), it ignores the given probability and computes the necessary `prob` to achieve a sample size close to `samplesize`

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 500))
S = distsample_ut(dist, X; samplesize=1000)  # ~1000 sampled pairwise distances
```
"""
function distsample_ut(dist::SemiMetric, X::AbstractDatabase; prob::Float64=0.01, samplesize=0)
    n = length(X)
    S = Float32[]
    if samplesize > 0
        prob = 2 * samplesize / n^2
        sizehint!(S, samplesize)
    else
        sizehint!(S, ceil(Int, 0.5 * prob * n^2))
    end

    for i = 1:n
        for j = (i+1):(n-1)
            if rand() <= prob
                push!(S, evaluate(dist, X[i], X[j]))
            end
        end
    end

    S
end

"""
    distsample(dist::PreMetric, X::AbstractDatabase; samplesize=ceil(Int, sqrt(length(X)))) -> S

Computes a sample of the pairwise distance matrix by drawing `samplesize` random pairs (with repetition,
possibly including an object paired with itself) from `X` and evaluating `dist` on each pair.
Returns an array of size `samplesize`.

# Arguments
- `dist`: distance function
- `X`: input database

# Keyword Arguments
- `samplesize`: the size of the sample

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 500))
S = distsample(dist, X)          # samplesize defaults to ceil(Int, sqrt(500))
S2 = distsample(dist, X; samplesize=1000)
```
"""
function distsample(dist::PreMetric, X::AbstractDatabase; samplesize=ceil(Int, sqrt(length(X))))
    n = length(X)
    S = Vector{Float32}(undef, samplesize)

    @BATCH minbatch=getminbatch(samplesize) for i in 1:samplesize
        u, v = rand(1:n), rand(1:n)
        S[i] = evaluate(dist, X[u], X[v])
    end

    S
end
