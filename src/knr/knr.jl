# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Dates

import SimilaritySearch:
    search, fit, push!, optimize!

export Knr

mutable struct Knr{T} <: Index
    db::Vector{T}
    refs::Vector{T}
    k::Int
    ksearch::Int
    minmatches::Int
    invindex::Vector{Vector{Int32}}
    verbose::Bool
end

"""
    fit(::Type{Knr}, dist, db::AbstractVector{T}, refs::AbstractVector{T}, k::Int, minmatches::Int=1, verbose=false) where T
    fit(::Type{Knr}, dist, db::AbstractVector{T}; numrefs::Int=1024, k::Int=3, minmatches::Int=1, tournamentsize::Int=3, verbose=false) where T

Creates a `Knr` index using the given references or the number of references to be used.

- `dist`: the distance function to be used
- `db`: the dataset to bbe indexed
- `refs`: the references to be used
- `k`: the number of references for representing objects inside the index
- `ksearch`: the same than `k` but used while searching
- `minmatches`: at query time, it determines the minimum number of references matched to allow a distance evaluation
- `verbose`: controls if the index must have a verbose output for its operations
- `numrefs`: if `refs` is not given, then `numrefs` objects are selected from `db` as `refs`
- `tournamentsize`: `numrefs` is specified, an incremental construction of `refs` is performed, each reference is selected as the more distant (to already selected references) among `tournamentsize` candidates
"""
function fit(::Type{Knr}, dist, db::AbstractVector{T}, refs::AbstractVector{T}, k::Int, minmatches::Int=1; verbose=false) where T
    verbose && println(stderr, "Knr> refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)")
    m = length(refs)
    invindex = [Vector{Int32}(undef, 0) for i in 1:m]
    seqindex = fit(Sequential, refs)
    counter = 0
    n = length(db)
    for i in 1:n
        res = search(seqindex, dist, db[i], KnnResult(k))
        for p in res
            push!(invindex[p.id], i)
        end
        counter += 1
        if (counter % 100_000) == 1
            println(stderr, "*** advance ", counter, " from ", n, "; ", string(Dates.now()))
        end
    end

    Knr(db, refs, k, k, minmatches, invindex, verbose)
end

function parallel_fit(::Type{Knr}, dist, db::AbstractVector{T}, refs::AbstractVector{T}, k::Int, minmatches::Int=1; verbose=false) where T
    verbose && println(stderr, "Knr> parallel_fit refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)")
    m = length(refs)
    invindex = [Vector{Int32}(undef, 0) for i in 1:m]
    locks = [Threads.SpinLock() for i in 1:m]
    seqindex = fit(Sequential, refs)
    counter = Threads.Atomic{Int}(0)
    n = length(db)
    Threads.@threads for i in 1:n
        res = search(seqindex, dist, db[i], KnnResult(k))
        for p in res
            refID = p.id
            lock(locks[refID])
            push!(invindex[refID], i)
            unlock(locks[refID])
        end
        Threads.atomic_add!(counter, 1)
        c = counter[]
        if (c % 100_000) == 1
            println(stderr, "*** advance ", c, " from ", n, "; ", string(Dates.now()))
        end
    end

    Knr(db, refs, k, k, minmatches, invindex, verbose)
end

function parallel_fit(::Type{Knr}, dist, db::AbstractVector{T}; numrefs::Int=1024, k::Int=3, minmatches::Int=1, tournamentsize::Int=3, verbose=false) where T
    # refs = rand(db, numrefs)
    refs = [db[x] for x in select_tournament(dist, db, numrefs, tournamentsize)]
    parallel_fit(Knr, dist, db, refs, k, minmatches, verbose=verbose)
end

function fit(::Type{Knr}, dist, db::AbstractVector{T}; numrefs::Int=1024, k::Int=3, minmatches::Int=1, tournamentsize::Int=3, verbose=false) where T
    # refs = rand(db, numrefs)
    refs = [db[x] for x in select_tournament(dist, db, numrefs, tournamentsize)]
    fit(Knr, dist, db, refs, k, minmatches, verbose=verbose)
end

"""
    search(index::Knr, dist, q, res::KnnResult)

Solves the query specified by `q` and `res` using the `Knr` index
"""
function search(index::Knr, dist, q, res::KnnResult)
    dz = zeros(Int16, length(index.db))
    # M = BitArray(length(index.db))
    seqindex = fit(Sequential, index.refs)
    kres = search(seqindex, dist, q, KnnResult(index.ksearch))

    for p in kres
        @inbounds for objID in index.invindex[p.id]
            c = dz[objID] + 1
            dz[objID] = c

            if c == index.minmatches
                d = dist(q, index.db[objID])
                push!(res, objID, d)
            end
        end
    end

    return res
end

"""
    push!(index::Knr, dist, obj)

Inserts `obj` into the index
"""
function push!(index::Knr, dist, obj)
    push!(index.db, obj)
    seqindex = fit(Sequential, index.refs)
    res = search(seqindex, dist, obj, KnnResult(index.k))
    for p in res
        push!(index.invindex[p.id], length(index.db))
    end
    
    length(index.db)
end

"""
    optimize!(index::Knr, dist; recall=0.9, k=10, num_queries=128, perf=nothing)

Optimizes the index to achieve the specified recall.
"""
function optimize!(index::Knr, dist; recall=0.9, k=10, num_queries=128, perf=nothing)
    index.verbose && println(stderr, "Knr> optimizing index for recall=$(recall)")
    if perf == nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end
    index.minmatches = 1
    index.ksearch = 1
    p = probe(perf, index, dist)

    while p.recall < recall && index.ksearch < length(index.refs)
        index.ksearch += 1
        index.verbose && println(stderr, "Knr> opt step ksearch=$(index.ksearch), performance $(p)")
        p = probe(perf, index, dist)

    end
    
    index.verbose && println(stderr, "Knr> reached performance $(p)")
    index
end