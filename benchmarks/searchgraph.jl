using SimilaritySearch, Random, JSON


generate_dataset(dim, n) = [rand(Float32, dim) for i in 1:n]


function main_search_graph(perf, S, k; automatic_optimization, opts...)
    println("==============  SearchGraph ================")
    println("=== objects: $(length(S)), dim=$(length(S[1])), knn: $k")

    start = time()
    G = SearchGraph(; opts...)
    if !automatic_optimization
        delete!(G.callback_list, :optimize_parameters)
    end

    println(G)
    append!(G, S)
    buildtime = time() - start 
    
    p = probe(perf, G)
    println(string(G))
    println("=== buildtime: $buildtime, queriespersecond: $(1/p.searchtime), recall: $(p.macrorecall)")
    println(JSON.json(p))
end

function main()
    k = 7
    automatic_optimization = false
    n = 100_000
    dist = SqL2Distance()
    nqueries = 1000
    for dim in [4, 8, 16]
        S = generate_dataset(dim, n)
        Q = generate_dataset(dim, nqueries)
        gold = ExhaustiveSearch(dist, S; ksearch=k)
        perf = Performance(gold, Q, k; popnearest=false)

        for salgo in [BeamSearch(4, 4)]
            for nalgo in [LogNeighborhood()]
                main_search_graph(perf, S, k;
                    dist=dist,
                    search_algo=salgo,
                    neighborhood_algo=nalgo,
                    automatic_optimization=automatic_optimization,
                    verbose=false
                )
            end
        end
    end
end

main()