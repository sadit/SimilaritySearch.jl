"""
    create_fitness_function(expected_recall::Float64; step::Float64=0.99, decrease::Float64=0.5, numsteps::Int=20)

# Description
Returns a fast discrete-sigmoid-like function with the desired behaviour

- `expected_recall` determines the objective recall (the desired top-inflection point of the sigmoid)
- `step` determines the `recall`'s decay to apply decrease
- `decrease` the decrease factor of the output on each step
- `numsteps` maximum number of steps to be applied
"""
function create_fitness_function(expected_recall::Float64; step::Float64=0.99, decrease::Float64=0.5, numsteps::Int=20)
    function fitness(p::PerformanceResult)
        recall = expected_recall
        # speed = 1.0 / p.distances
        speed = 1.0 / p.seconds

        for i in 1:numsteps
            if p.recall >= recall
                return 1.0 + p.recall + speed
            end

            recall *= step
            speed *= decrease
        end

        return p.recall
    end

    return fitness
end

"""
    optimize_algo!(algosearch::S, index::LocalSearchIndex{T}, recall::Float64, perf::Performance) where {T, S <: LocalSearchAlgorithm}

Optimizes a local search index for an specific algorithm to get the desired recall. Note that optimize for low-recall will yield to faster searches.
The train queries are specified as part of the `perf` struct.
"""
function optimize_algo!(algosearch::S, index::LocalSearchIndex{T}, recall::Float64, perf::Performance) where {T, S <: LocalSearchAlgorithm}
    n = length(index.db)
    optimize_neighborhood!(index.neighborhood_algo, index, perf, recall)
    tabu = Set{S}()
    candidates_population = 2  ## a magic number
    candidates = Vector{Tuple{Float64,S}}()
    best_perf = probe(perf, index)
    fitness_function = create_fitness_function(recall)

    # push!(candidates, (fitness_performance(recall, best_perf), algosearch))
    push!(candidates, (fitness_function(best_perf), algosearch))
    push!(tabu, algosearch)
    index.options.verbose && info("XXX $(typeof(algosearch)). Starting parameter optimization; expected recall: $recall, n: $n")
    best_fitness, best_state = candidates[end]
    iter = 0
    while length(candidates) > 0
        iter += 1
        prev_fitness, prev_state = pop!(candidates)
        index.options.verbose && info("XXX $(typeof(algosearch)). Iteration: $iter, expected recall: $recall, n: $n")
        # for s in @task opt_expand_neighborhood(prev_state, n, iter)
        opt_expand_neighborhood(prev_state, n, iter) do s
            if !in(s, tabu)
                push!(tabu, s)
                index.search_algo = s

                # p = probe(perf, index, repeat=3, aggregation=:median, field=:seconds)
                p = probe(perf, index, repeat=1, field=:seconds)
                fitness = fitness_function(p)
                push!(candidates, (fitness, s))
                if fitness > best_fitness
                    best_fitness, best_state, best_perf = fitness, s, p
                    index.options.verbose && info("*** $(typeof(algosearch)). A new best conf was found> fitness: $fitness, conf: $(JSON.json(s)), perf: $(JSON.json(p)), candidates: $(length(candidates)), n: $(n)")
                end
            end
        end

        if length(candidates) > 0
            sort!(candidates, by=(x) -> x[1])
            sp = max(1, length(candidates)-candidates_population+1)
            candidates = candidates[sp:end]
            candidates = filter((x) -> x[1] >= best_fitness, candidates) |> collect
            index.options.verbose && info("=== $(typeof(algosearch)). Iteration finished; fitness: $(JSON.json(best_fitness)), conf: $(JSON.json(best_state)), perf: $(JSON.json(best_perf)) candidates: $(length(candidates)), n: $(n)")
        end
    end


    #for field in fieldnames(algosearch)
    #    setfield!(algosearch, field, getfield(best_state, field))
    #end
    index.search_algo = best_state

    index.options.verbose && info("XXX $(typeof(algosearch)). Optimization done; fitness: $(JSON.json(best_fitness)), conf: $(JSON.json(best_state)), perf: $(JSON.json(best_perf)), n: $(n)")
    return index
end



