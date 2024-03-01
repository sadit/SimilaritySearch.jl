# This file is part of SimilaritySearch.jl
export distsample_ut, distsample

"""
    distsample_ut(dist::SemiMetric, X::AbstractDatabase; prob=0.01, samplesize=0)
    distsample(dist::SemiMetric, X::AbstractDatabase; prob=0.01, samplesize=sqrt(|X|))
    
    Computes a sample of the upper triangular pairwise distance matrix. 
    Returns an array of distances of close to ``prob * n^2/2`` entries for a database of size ``n``.
    This method is fine to work in small datasets (not million sized datasets); this method do not return duplicates nor symmetric duplicates
    
    - `dist`: Distance function
    - `X`: input database
    - `prob`: sampling probability (on the upper triangle pairwise distance matrix)
    - `samplesize`: if samplesize is given, the it ignores the given probability and computes the necessary `prob` to achieve a value close to `samplesize` 
"""
function distsample_ut(dist::SemiMetric, X::AbstractDatabase; prob::Float64 = 0.01, samplesize=0)
    n = length(X)
    S = Float32[]
    if samplesize > 0
        prob = 2 * samplesize / n^2
        sizehint!(S, samplesize)
    else
        sizehint!(S, ceil(Int, 0.5 * prob * n^2))
    end

    for i = 1:n
        for j = (i + 1):(n - 1)
            if rand() <= prob
                push!(S, evaluate(dist, X[i], X[j]))
            end
        end
    end

    S
end

"""
    distsample(dist::SemiMetric, X::AbstractDatabase; samplesize=sqrt(|X|))
    
    Computes a sample of the pairwise distance matrix. 
    Returns anarray of size `samplesize`
    
    - `dist`: Distance function
    - `X`: input database
    - `samplesize`: the size of the sample
"""
function distsample(dist::SemiMetric, X::AbstractDatabase; samplesize=ceil(Int, sqrt(length(X))))
    n = length(X)
    S = Vector{Float32}(undef, samplesize)

    Threads.@threads :static for i in 1:samplesize
        u, v = rand(1:n), rand(1:n) 
        S[i] = evaluate(dist, X[u], X[v])
    end

    S
end
