# This file is part of SimilaritySearch.jl

"""
    beamsearchsat!(sat, ctx, root, bs, q, res; tabu=zero(UInt32)) -> res

Approximate SAT traversal seeded at `root`. `tabu` (used by [`BeamSearchParSat`](@ref)'s
ancestor climb) excludes one child id from being (re-)evaluated. Mutates `res`, returns it,
and accounts distance evaluations into `ctx`.
"""
function beamsearchsat!(sat::Sat, ctx::SatContext, root::UInt32, bs::BeamSearch, q, res::AbstractKnnQueue;
                         tabu::UInt32=zero(UInt32))
    dist = distance(sat)
    cost = 1
    d = Dist.evaluate(dist, q, database(sat, root))
    push_item!(res, root, d)

    beam = getbeam(bs.bsize, ctx)
    sat.children[root] !== nothing && push_item!(beam, root, d)
    Δ = bs.Δ

    @inbounds while length(beam) > 0
        prev = pop_min!(beam)
        C = sat.children[prev.id]::Vector{UInt32}
        for c in C
            c == tabu && continue
            d = Dist.evaluate(dist, q, database(sat, c))
            cost += 1
            push_item!(res, c, d)

            if sat.children[c] !== nothing && d <= Δ * covradius(res) + sat.cov[c]
                push_item!(beam, c, d)
            end
        end
    end

    add_distance_evaluations!(ctx, cost)
    res
end

struct BeamSearchSat{ST<:Sat} <: AbstractSearchIndex
    algo::Ref{BeamSearch}
    sat::ST
end

"""
    BeamSearchSat(sat::Sat; bsize=8, Δ=1f0, maxvisits=typemax(Int))

Approximate SAT search via aggressive beam pruning; adapted from Chávez & Navarro (2001).
Autotunes via [`optimize_index!`](@ref) using the existing `BeamSearchSpace`/`combine`/
`mutate` hooks already defined for `BeamSearch` (`SearchGraph`'s tuning machinery).
"""
BeamSearchSat(sat::Sat; bsize::Integer=8, Δ::Real=1f0, maxvisits::Integer=typemax(Int)) =
    BeamSearchSat(Ref(BeamSearch(; bsize, Δ, maxvisits)), sat)

@inline database(b::BeamSearchSat) = database(b.sat)
@inline database(b::BeamSearchSat, i) = database(b.sat, i)
@inline distance(b::BeamSearchSat) = distance(b.sat)
@inline Base.length(b::BeamSearchSat) = length(b.sat)

function Base.show(io::IO, b::BeamSearchSat)
    println(io, typeof(b), " algo=", b.algo[])
    show(io, b.sat)
end

search(b::BeamSearchSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    beamsearchsat!(b.sat, ctx, b.sat.root, b.algo[], q, res)

## optimize_index! integration -- reuses SearchGraph's BeamSearchSpace/combine/mutate as-is
optimization_space(::BeamSearchSat) = BeamSearchSpace(;
    bsize = 8:16:64, Δ = [0.8, 1.0, 1.3, 1.5],
    bsize_scale = (s=1.5, p1=0.5, p2=0.5, lower=4, upper=256),
    Δ_scale = (s=1.2, p1=0.5, p2=0.5, lower=0.5, upper=3.0))

setconfig!(bs::BeamSearch, index::BeamSearchSat, perf) = (index.algo[] = bs)

runconfig(bs::BeamSearch, index::BeamSearchSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    beamsearchsat!(index.sat, ctx, index.sat.root, bs, q, res)
