# function fitness_performance(expected_recall::Float64, p::PerformanceResult)
#     # computes the fitness of a given performance
#     # expected_recall determines the minimum recall after reaching expected_recall,
#     # the number of queries per second becomes the major parameter
#     # a = 10.0 / (1.0 - expected_recall)  ## controlling the slope, put it in terms of the expected recall to obtain better approximations as the expected recall is near to 1.0
#     # c = 1.0 / (1.0 + exp(-a*(p.recall-expected_recall)))
#     # c = 1.0 / (1.0 + exp(-a*(p.recall-expected_recall)))
#     # return p.recall + 1.0/ p.seconds * c
#     # return min(p.recall, expected_recall) + 1.0/ p.distances * c
#     # return (1 - c) * p.recall + c / p.seconds
#     # return (1 - c) * p.recall + c / p.distances
#     if p.recall < expected_recall
#         return p.recall
#     else
#         return 1 + 1.0 / p.seconds
#         # return 10.0 + 1.0 / p.distances
#     end
# end

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

function optimize_algo!{T, S <: LocalSearchAlgorithm}(algosearch::S, index::LocalSearchIndex{T}, recall::Float64, perf::Performance)
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
                #for field in fieldnames(algosearch)
                #    setfield!(algosearch, field, getfield(s, field))
                #end
                index.search_algo = s

                p = probe(perf, index, repeat=3, aggregation=:median, field=:seconds)
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


# function dominates(a::PerformanceResult, b::PerformanceResult)
#     # speedup =>    1.0 / a.seconds >= 1 / b.seconds
#     return a.recall >= b.recall && a.seconds <= b.seconds
#     # return a.recall >= b.recall && a.distances < b.distances
# end

# function dominates(minimum_recall::Float64, a::PerformanceResult, b::PerformanceResult)
#     # determines if `a` performance dominates `b`; after minimum_recall is reached the search time determines the domination
#     return fitness_performance(minimum_recall, a) > fitness_performance(minimum_recall, b)
#     # dominates(a, b)
#     # if a.recall < minimum_recall || b.recall < minimum_recall
#     #     return performance_dominated(a, b)
#     # else
#     #     return a.seconds <= b.seconds
#     # end
#     # return performance_dominated(a, b)
# end
