using Dates

"""
    optimize!

Optimizes a local search index for an specific algorithm to get the desired performance.
Note that optimizing for low-recall will yield to faster searches; the train queries
are specified as part of the `perf` struct.
"""
function optimize!(algosearch::LocalSearchAlgorithm,
                   index::SearchGraph{T},
                   dist::Function,
                   recall::Float64,
                   perf::Performance;
                   bsize::Int=4,
                   tol::Float64=0.01) where T
    n = length(index.db)
    score_function(p) = p.recall < recall ? p.recall : 1.0 + n / p.evaluations
    #score_function(p) = p.recall < recall ? p.recall : 1.0 + 1.0 / (1.0 + sum(p.distances))
    #score_function(p) = 1.0 / (1.0 + sum(p.distances))
    optimize_neighborhood!(index.neighborhood_algo, index, dist, perf, recall)
    p = probe(perf, index, dist)
    best_list = [(score=score_function(p), state=algosearch, perf=p)]
    exploration = Dict(algosearch => 0)  ## -1 unexplored; 0 visited; 1 visited & expanded

    @info "==== BEGIN $(typeof(algosearch)). Starting parameter optimization; expected recall: $recall, n: $n"
    prev_score = -1.0
    iter = 0
    while abs(best_list[1].score - prev_score) > tol
        iter += 1
        prev_score = best_list[1].score
        @info "  == begin iteration $(typeof(algosearch)). Iteration: $iter, expected recall: $recall, n: $n"
        
        for prev in @view best_list[1:end]  ## the view also fixes the size of best_list even after push!
            S = get(exploration, prev.state, -1)
            if S == 1
                continue  # visited and explored
            elseif S == 0
                exploration[prev.state] = 1
            end

            opt_expand_neighborhood(prev.state, n, iter) do state
                S = get(exploration, state, -1)
                if S == -1
                    exploration[state] = 0
                    index.search_algo = state
                    # p = probe(perf, index, repeat=3, aggregation=:median, field=:seconds)
                    p = probe(perf, index, dist, repeat=1, field=:seconds)
                    score = score_function(p)
                    push!(best_list, (score=score, state=state, perf=p))
                    if score > best_list[1].score
                        @info "  ** $(typeof(algosearch)). A new best conf was found> score: $score, conf: $(JSON.json(state)), perf: $(JSON.json(p)), best_list's length: $(length(best_list)), n: $(n)"
                    end
                end
            end
        end
    
        sort!(best_list, by=(x) -> -x.score)
        if length(best_list) > bsize
            best_list = best_list[1:bsize]
        end
        @info "  == end $(typeof(algosearch)). Iteration finished; $(JSON.json(best_list[1])), beam: $(length(best_list)), n: $(n)"
    end

    index.search_algo = best_list[1].state
    @info "==== END $(typeof(algosearch)). Finished, best: $(JSON.json(best_list[1])), n: $(n)"
    return index
end



