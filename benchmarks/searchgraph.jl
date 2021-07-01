using SimilaritySearch, Random, JSON


generate_dataset(dim, n) = [rand(Float32, dim) for i in 1:n]

function main_search_graph(perf, S, k; optimize_parameters, opts...)
    println("==============  SearchGraph ================")
    println("=== objects: $(length(S)), dim=$(length(S[1])), knn: $k")

    start = time()
    G = SearchGraph(; opts...)
    !optimize_parameters && delete!(G.callbacks, :parameters)
    #G.neighborhood.logbase = 1.5
    #G.neighborhood.logbase = Inf
    #G.neighborhood.minsize = 10

    append!(G, S)
    buildtime = time() - start 
    
    p = probe(perf, G)
    println(string(G))
    println("=== buildtime: $buildtime, queriespersecond: $(1/p.searchtime), recall: $(p.macrorecall)")
    println(JSON.json(p))
end

function main()
    k = 7
    optimize_parameters = false
    n = 100_000
    dist = SqL2Distance()
    nqueries = 1000
    for dim in [4, 8, 16]
        S = generate_dataset(dim, n)
        Q = generate_dataset(dim, nqueries)
        gold = ExhaustiveSearch(dist, S; ksearch=k)
        perf = Performance(gold, Q, k; popnearest=false)

        for salgo in [BeamSearch(bsize=4)]
            main_search_graph(perf, S, k;
                dist=dist,
                search_algo=salgo,
                optimize_parameters=optimize_parameters,
                verbose=false
            )
        end
    end
end

main()
