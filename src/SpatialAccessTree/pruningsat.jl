# This file is part of SimilaritySearch.jl

struct PruningSatConfig
    factor::Float32
end

struct PruningSat{ST<:Sat} <: AbstractSearchIndex
    config::Ref{PruningSatConfig}
    sat::ST
end

"""
    PruningSat(sat::Sat; factor=0.9)

Aggressive-pruning approximate SAT search ("probabilistic spell", Chávez & Navarro 2001).
`factor==1` recovers exact search. Autotunes via [`optimize_index!`](@ref) using
[`PruningSatSpace`](@ref).
"""
PruningSat(sat::Sat; factor::Real=0.9) = PruningSat(Ref(PruningSatConfig(convert(Float32, factor))), sat)

@inline database(p::PruningSat) = database(p.sat)
@inline database(p::PruningSat, i) = database(p.sat, i)
@inline distance(p::PruningSat) = distance(p.sat)
@inline Base.length(p::PruningSat) = length(p.sat)

function Base.show(io::IO, p::PruningSat)
    println(io, typeof(p), " config=", p.config[])
    show(io, p.sat)
end

explore_node!(sat::Sat, ctx::SatContext, q, ::Nothing, res::AbstractKnnQueue, queue::Vector{UInt32}) = 0

function explore_node!(sat::Sat, ctx::SatContext, q, C::Vector{UInt32}, res::AbstractKnnQueue, queue::Vector{UInt32})
    cost = 0
    dist = distance(sat)
    for c in C
        if sat.children[c] === nothing
            d = Dist.evaluate(dist, q, database(sat, c))
            cost += 1
            push_item!(res, c, d)
        else
            push!(queue, c)
        end
    end

    cost
end

function pruningsearchtree!(sat::Sat, ctx::SatContext, q, p::UInt32, res::AbstractKnnQueue, factor::Float32)
    queue = getqueue(ctx)
    push!(queue, p)
    dist = distance(sat)
    cost = 0

    @inbounds while length(queue) > 0
        p = pop!(queue)
        dqp = Dist.evaluate(dist, q, database(sat, p))
        cost += 1
        push_item!(res, p, dqp)

        if dqp < factor * covradius(res) + sat.cov[p]
            cost += explore_node!(sat, ctx, q, sat.children[p], res, queue)
        end
    end

    add_distance_evaluations!(ctx, cost)
    res
end

search(p::PruningSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    pruningsearchtree!(p.sat, ctx, q, p.sat.root, res, p.config[].factor)

## optimize_index! integration

"""
    PruningSatSpace(; factor=0.2:0.1:0.9, factor_scale=(...)) <: AbstractSolutionSpace

Autotuning search space for [`PruningSat`](@ref)'s pruning `factor`.
"""
@kwdef struct PruningSatSpace <: AbstractSolutionSpace
    factor = 0.2:0.1:0.9
    factor_scale = (s=1.07, p1=0.5, p2=0.5, lower=0.1, upper=0.9999)
end

Base.hash(c::PruningSatConfig, h::UInt) = hash(round(c.factor, digits=4), h)
Base.isequal(a::PruningSatConfig, b::PruningSatConfig) = a.factor == b.factor
Base.eltype(::PruningSatSpace) = PruningSatConfig
Base.rand(rng::AbstractRNG, space::PruningSatSpace) = PruningSatConfig(rand(rng, space.factor))

combine(a::PruningSatConfig, b::PruningSatConfig) = PruningSatConfig((a.factor + b.factor) / 2)

mutate(space::PruningSatSpace, c::PruningSatConfig, iter) =
    PruningSatConfig(scale(c.factor; space.factor_scale...))

optimization_space(::PruningSat) = PruningSatSpace()
setconfig!(config::PruningSatConfig, index::PruningSat, perf) = (index.config[] = config)

runconfig(config::PruningSatConfig, index::PruningSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    pruningsearchtree!(index.sat, ctx, q, index.sat.root, res, config.factor)
