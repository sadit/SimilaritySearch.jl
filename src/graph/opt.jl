# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Dates

"""
    function optimize!(search_algo::LocalSearchAlgorithm,
                       index::SearchGraph{T},
                       recall::Float64,
                       perf::Performance;
                       bsize::Int=4,
                       tol::Float64=0.01,
                       probes::Int=0) where T

Optimizes a local search index for an specific algorithm to get the desired performance.
Note that optimizing for low-recall will yield to faster searches; the train queries
are specified as part of the `perf` struct.
"""
function optimize!(
        perf::Performance,
        search_algo::LocalSearchAlgorithm,
        index::SearchGraph;
        recall::Float64=0.9,
        bsize::Integer=4,
        tol::Float64=0.01,
        maxiters::Integer=3,
        probes::Integer=0
    )
    n = length(index.db)
    score_function(p) = p.macrorecall < recall ? p.macrorecall : 1.0 + n / p.evaluations
    # score_function(p) = p.macrorecall < recall ? p.macrorecall : 1.0 + n / p.evaluations
    #score_function(p) = p.recall < recall ? p.recall : 1.0 + 1.0 / (1.0 + sum(p.evaluations))
    #score_function(p) = 1.0 / (1.0 + sum(p.evaluations))
    p = probe(perf, index)
    best_list = [(score=score_function(p), state=search_algo, perf=p)]
    exploration = Dict(search_algo => 0)  ## -1 unexplored; 0 visited; 1 visited & expanded

    index.opts.verbose && println(stderr, "==== BEGIN Opt. $(string(search_algo)), expected recall: $recall, n: $n")
    prev_score = -1.0
    iter = 0

    state_string(s) = "(state=$(string(s.state)), score=$(s.score), perf=$(s.perf)  )"
    while abs(best_list[1].score - prev_score) > tol && iter < maxiters
        iter += 1
        prev_score = best_list[1].score
        index.opts.verbose && println(stderr, "  == Begin Opt. $(string(search_algo)) iteration: $iter, expected recall: $recall, n: $n")
        
        for prev in @view best_list[1:end]  ## the view also fixes the size of best_list even after push!
            S = get(exploration, prev.state, -1)
            if S == 1
                continue  # visited and explored
            elseif S == 0
                exploration[prev.state] = 1
            end
            
            opt_expand_neighborhood(prev.state, n, iter, probes) do state
                S = get(exploration, state, -1)
                if S == -1
                    exploration[state] = 0
                    copy!(index.search_algo, state)
                    p = probe(perf, index)
                    score = score_function(p)
                    if length(best_list) < bsize || score > best_list[bsize].score
                        push!(best_list, (score=score, state=state, perf=p))
                        if score > best_list[1].score
                            index.opts.verbose && println(stderr, "    ** new best conf: $(state_string(best_list[end])), #beam: $(length(best_list)), n: $(n)")
                        end
                    end
                end
            end
        end
    
        sort!(best_list, by=(x) -> -x.score)
        if length(best_list) > bsize
            best_list = best_list[1:bsize]
        end
        index.opts.verbose && println(stderr, "  == Iteration finished, current best: $(state_string(best_list[1])), #beam: $(length(best_list)), n: $(n)")
    end

    copy!(index.search_algo, best_list[1].state)
    index.opts.verbose && println(stderr, "==== END Opt. Finished, best: $(state_string(best_list[1])), n: $(n)")
    index
end



