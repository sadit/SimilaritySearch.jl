# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Dates
export Knr, KnrOptions

mutable struct KnrOptions
    ksearch::Int32
    minmatches::Int32
    verbose::Bool
end

struct Knr{RefSearchType<:AbstractSearchContext, DataType<:AbstractVector, DistanceType<:PreMetric} <: AbstractSearchContext
    dist::DistanceType
    db::DataType
    refsearch::RefSearchType
    kbuild::Int32
    invindex::Vector{Vector{Int32}}
    res::KnnResult
    opts::KnrOptions
end

Base.copy(knr::Knr; dist=knr.dist, db=knr.db, refsearch=knr.refsearch, kbuild=knr.kbuild, invindex=knr.invindex, res=knr.res, opts=knr.opts) =
    Knr(dist, db, refsearch, kbuild, invindex, res, opts)

function Knr(
    dist::PreMetric,
    db::AbstractVector,
    refsearch::AbstractSearchContext,
    kbuild::Integer,
    invindex;
    ksearch::Integer=kbuild,
    minmatches::Integer=1,
    verbose=true,
    k::Integer=10
)
    kbuild = convert(Int32, kbuild)
    minmatches = convert(Int32, minmatches)
    minmatches = min(kbuild, minmatches)
    opts = KnrOptions(convert(Int32, ksearch), minmatches, verbose)
    Knr(dist, db, refsearch, kbuild, invindex, KnnResult(k), opts)
end

"""
    Knr(dist::PreMetric, db::AbstractVector{T}, refsearch::AbstractSearchContext;
        kbuild::Integer=3,
        ksearch::Integer=kbuild,
        minmatches::Integer=1,
        verbose=true,
        k::Integer=10)
    Knr(dist::PreMetric, db::AbstractVector{T};
        numrefs::Int=1024,
        kbuild::Integer=3,
        ksearch::Integer=kbuild,
        minmatches::Integer=1,
        tournamentsize::Int=3,
        verbose=false,
        k=10) where T

Creates a `Knr` index using the given references or the number of references to be used.

- `dist`: the distance function to be used (a PreMetric object as described in `Distances.jl`)
- `db`: the dataset to bbe indexed
- `refsearch`: the references index to be used
- `kbuild`: the number of references for representing objects inside the index
- `ksearch`: the same than `k` but used while searching
- `minmatches`: at query time, it determines the minimum number of references matched to allow a distance evaluation
- `verbose`: controls if the index must have a verbose output for its operations
- `numrefs`: if `refs` is not given, then `numrefs` objects are selected from `db` as `refs`
- `tournamentsize`: `numrefs` is specified, an incremental construction of `refs` is performed, each reference is selected as the more distant (to already selected references) among `tournamentsize` candidates
"""
function Knr(dist::PreMetric, db::AbstractVector{T}, refsearch::AbstractSearchContext;
    kbuild::Integer=3, ksearch::Integer=kbuild, minmatches::Integer=1, verbose=true, k::Integer=10) where T

    verbose && println(stderr, "Knr> refs=$(typeof(db)), k=$(kbuild), numrefs=$(length(refs)), dist=$(dist)")
    m = length(refsearch.db)
    invindex = [Vector{Int32}(undef, 0) for i in 1:m]
    counter = 0
    n = length(db)

    for i in 1:n
        for p in search(refsearch, db[i], kbuild)
            push!(invindex[p.id], i)
        end

        counter += 1
        if verbose && (counter % 100_000) == 1
            println(stderr, "*** advance ", counter, " from ", n, "; ", string(Dates.now()))
        end
    end

    verbose && println(stderr, "*** advance ", counter, " from ", n, "; ", string(Dates.now()))
    Knr(dist, db, refsearch, kbuild, invindex; ksearch=ksearch, minmatches=minmatches, verbose=verbose, k=k)
end

function Knr(dist::PreMetric, db::AbstractVector{T};
        numrefs::Int=1024,
        kbuild::Integer=3,
        ksearch::Integer=kbuild,
        minmatches::Integer=1,
        tournamentsize::Int=3,
        verbose=false,
        k=10) where T
    # refs = rand(db, numrefs)
    refs = db[select_tournament(dist, db, numrefs, tournamentsize)]
    refsearch = ExhaustiveSearch(dist, refs; ksearch=kbuild)
    Knr(dist, db, refsearch; kbuild=kbuild, ksearch=ksearch, minmatches=minmatches, verbose=verbose, k=k)
end

## """
##     parallel_Knr(dist::PreMetric, db::AbstractVector{T}, refs::AbstractVector{T}, k::Int, minmatches::Int=1; verbose=false) where T
## 
## Create a Knr index in parallel using the available threads. 
## """
## function parallel_Knr(dist::PreMetric, db::AbstractVector{T}, refs::AbstractVector{T}, k::Int, minmatches::Int=1; verbose=false) where T
##     verbose && println(stderr, "Knr> parallel_fit refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)")
##     m = length(refs)
##     invindex = [Vector{Int32}(undef, 0) for i in 1:m]
##     locks = [Threads.SpinLock() for i in 1:m]
##     refsctx = MultithreadedSequentialSearchContext(refs, dist, k)
##     counter = Threads.Atomic{Int}(0)
##     n = length(db)
## 
##     Threads.@threads for i in 1:n
##         res = search(refsctx, db[i])
##         for p in res
##             refID = p.id
##             lock(locks[refID])
##             push!(invindex[refID], i)
##             unlock(locks[refID])
##         end
## 
##         Threads.atomic_add!(counter, 1)
##         c = counter[]
##         if (c % 100_000) == 1
##             println(stderr, "*** advance ", c, " from ", n, "; ", string(Dates.now()))
##         end
##     end
## 
##     Knr(refs, k, k, minmatches, invindex, verbose)
## end
## 
## function parallel_fit(::Type{Knr}, dist::PreMetric, db::AbstractVector{T}; numrefs::Int=1024, k::Int=3, minmatches::Int=1, tournamentsize::Int=3, verbose=false) where T
##     # refs = rand(db, numrefs)
##     refs = [db[x] for x in select_tournament(dist, db, numrefs, tournamentsize)]
##     parallel_fit(Knr, dist, db, refs, k, minmatches, verbose=verbose)
## end

"""
    search(knr::Knr, q, res::KnnResult)

Solves the query specified by `q` and `res` using the `Knr` index
"""
function search(knr::Knr, q, res::KnnResult)
    # dz = zeros(Int16, length(knr.db))
    dz = Dict{Int32,Int8}()
    kres = search(knr.refsearch, q, knr.opts.ksearch)
    minmatches = knr.opts.minmatches

    @inbounds for i in eachindex(kres)
        p = kres[i]
        for objID in knr.invindex[p.id]
            #c = dz[objID] + 1
            #dz[objID] = c
            c = get(dz, objID, zero(Int8)) + one(Int8)
            dz[objID] = c
            if c == minmatches
                d = evaluate(knr.dist, q, knr.db[objID])
                push!(res, objID, d)
            end
        end
    end

    res
end

"""
    optimize!(perf::Performance, index::Knr; recall=0.9, ksearch=10, verbose=index.opts.verbose)


Optimizes the index to achieve the specified recall.
"""
function optimize!(perf::Performance, index::Knr; recall=0.9, ksearch=10, verbose=index.opts.verbose)
    verbose && println(stderr, "Knr> optimizing index for recall=$(recall)")
    index.opts.minmatches = 1
    index.opts.ksearch = 1
    p = probe(perf, index)

    while p.macrorecall < recall
        index.opts.ksearch += 1
        verbose && println(stderr, "Knr> opt step ksearch=$(index.ksearch), performance $(p)")
        p = probe(perf, index)
    end
    
    verbose && println(stderr, "Knr> reached performance $(p)")
    index
end

## """
##     push!(index::Knr, dist::PreMetric, obj)
## 
## Inserts `obj` into the index
## """
## function push!(index::Knr, dist::PreMetric, obj)
##     push!(index.db, obj)
##     seqindex = fit(Sequential, index.refs)
##     res = search(seqindex, dist, obj, KnnResult(index.k))
##     for p in res
##         push!(index.invindex[p.id], length(index.db))
##     end
##     
##     length(index.db)
## end
