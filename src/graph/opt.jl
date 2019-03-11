using Dates

"""
    create_score_function(expected_recall::Float64; step::Float64=0.99, decrease::Float64=0.5, numsteps::Int=20)

# Description
Returns a fast discrete-sigmoid-like function with the desired behaviour

- `expected_recall` determines the objective recall (the desired top-inflection point of the sigmoid)
- `step` determines the `recall`'s decay to apply decrease
- `decrease` the decrease factor of the output on each step
- `numsteps` maximum number of steps to be applied
"""
function create_score_function(expected_recall::Float64; step::Float64=0.99, decrease::Float64=0.5, numsteps::Int=20)
    function score_fun(p::PerformanceResult)
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

        p.recall
    end

    score_fun
end

"""
    optimize_algo!

Optimizes a local search index for an specific algorithm to get the desired recall. Note that optimize for low-recall will yield to faster searches.
The train queries are specified as part of the `perf` struct.
"""
function optimize_algo!(algosearch::LocalSearchAlgorithm,
                        index::SearchGraph{T},
                        dist::Function,
                        recall::Float64,
                        perf::Performance;
                        bsize::Int=4,
                        tol::Float64=0.01) where T
    n = length(index.db)
    optimize_neighborhood!(index.neighborhood_algo, index, dist, perf, recall)
    p = probe(perf, index, dist)
    score_function = create_score_function(recall)
    best_list = [(score=score_function(p), state=algosearch, perf=p)]
    exploration = Dict(algosearch => 0)  ## -1 unexplored; 0 visited; 1 visited & expanded

    @debug "XXX $(typeof(algosearch)). Starting parameter optimization; expected recall: $recall, n: $n"
    prev_score = -1.0
    iter = 0
    while abs(best_list[1].score - prev_score) > tol
        iter += 1
        prev_score = best_list[1].score
        @debug "XXX $(typeof(algosearch)). Iteration: $iter, expected recall: $recall, n: $n"
        
        for prev in @view best_list[1:end]  ## the view also fixes the size of best_list even after push!
            S = get(exploration, prev.state, -1)
            if S == 1
                continue  # visited and explored
            elseif S == 0
                exploration[prev.state] = 1
            end
                # @debug "XXX--- $(typeof(algosearch)). Iteration: $iter, expected recall: $prev, n: $n"
            opt_expand_neighborhood(prev.state, n, iter) do state
                S = get(exploration, state, -1)
                if S == -1
                    exploration[state] = 0
                    index.search_algo = state
                    # p = probe(perf, index, repeat=3, aggregation=:median, field=:seconds)
                    p = probe(perf, index, dist, repeat=1, field=:seconds)
                    score = score_function(p)
                    push!(best_list, (score=score, state=state, perf=p))
                    if score > best_list[end].score
                        @debug "*** $(typeof(algosearch)). A new best conf was found> score: $score, conf: $(JSON.json(state)), perf: $(JSON.json(p)), best_list: $(length(best_list)), n: $(n)"
                    end
                end
            end
        end

        sort!(best_list, by=(x) -> -x.score)
        if length(best_list) > bsize
            best_list = best_list[1:bsize]
        end
        @debug "=== $(typeof(algosearch)). Iteration finished; $(JSON.json(best_list[1])), beam: $(length(best_list)), n: $(n)"
    end

    index.search_algo = best_list[1].state
    @debug "XXX $(typeof(algosearch)). Optimization done. $(JSON.json(best_list[1])), n: $(n)"
    return index
end



