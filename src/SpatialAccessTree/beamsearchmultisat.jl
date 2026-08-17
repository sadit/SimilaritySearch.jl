# This file is part of SimilaritySearch.jl

struct BeamSearchMultiSat{ST<:Sat} <: AbstractSearchIndex
    algo::Ref{BeamSearch}
    sat::Vector{ST}
end

"""
    BeamSearchMultiSat(sat::Vector{<:Sat}; bsize=8, Δ=1f0, maxvisits=typemax(Int))

Beam search over a forest of `Sat` partitions of the *same* underlying database (e.g. built
via `RandomInitialPartition`). A node internal in *any* tree of the forest is expanded
through all trees where it is internal, increasing the chance of finding a good descent
path even when it's a leaf in some particular tree.
"""
BeamSearchMultiSat(sat::Vector{ST}; bsize::Integer=8, Δ::Real=1f0, maxvisits::Integer=typemax(Int)) where {ST<:Sat} =
    BeamSearchMultiSat(Ref(BeamSearch(; bsize, Δ, maxvisits)), sat)

@inline database(b::BeamSearchMultiSat) = database(b.sat[1])
@inline database(b::BeamSearchMultiSat, i) = database(b.sat[1], i)
@inline distance(b::BeamSearchMultiSat) = distance(b.sat[1])
@inline Base.length(b::BeamSearchMultiSat) = length(b.sat[1])

function Base.show(io::IO, b::BeamSearchMultiSat)
    println(io, typeof(b), " algo=", b.algo[], " forest-size=", length(b.sat))
    show(io, b.sat[1])
end

function beamsearchmultisat!(satarr::Vector{ST}, ctx::SatContext, bs::BeamSearch, q, res::AbstractKnnQueue) where {ST<:Sat}
    sat1 = satarr[1]
    dist = distance(sat1)
    n = length(sat1)
    vstate = getvstate(n, ctx)
    root = sat1.root
    cost = 1

    d = Dist.evaluate(dist, q, database(sat1, root))
    push_item!(res, root, d)
    visit!(vstate, convert(UInt64, root))

    beam = getbeam(bs.bsize, ctx)
    for sat in satarr
        if sat.children[root] !== nothing
            push_item!(beam, root, d)
            break
        end
    end

    Δ = bs.Δ
    @inbounds while length(beam) > 0
        prev = pop_min!(beam)
        for sat in satarr                      # expand through every tree where prev is internal
            C = sat.children[prev.id]
            C === nothing && continue
            for c in C
                check_visited_and_visit!(vstate, convert(UInt64, c)) && continue
                d = Dist.evaluate(dist, q, database(sat, c))
                cost += 1
                push_item!(res, c, d)

                isinternal = any(s -> s.children[c] !== nothing, satarr)
                if isinternal && d <= Δ * covradius(res)
                    push_item!(beam, c, d)
                end
            end
        end
    end

    add_distance_evaluations!(ctx, cost)
    res
end

search(b::BeamSearchMultiSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    beamsearchmultisat!(b.sat, ctx, b.algo[], q, res)

optimization_space(::BeamSearchMultiSat) = BeamSearchSpace(;
    bsize = 8:16:64, Δ = [0.8, 1.0, 1.3, 1.5],
    bsize_scale = (s=1.5, p1=0.5, p2=0.5, lower=4, upper=256),
    Δ_scale = (s=1.1, p1=0.5, p2=0.5, lower=0.5, upper=3.0))

setconfig!(bs::BeamSearch, index::BeamSearchMultiSat, perf) = (index.algo[] = bs)

runconfig(bs::BeamSearch, index::BeamSearchMultiSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    beamsearchmultisat!(index.sat, ctx, bs, q, res)
