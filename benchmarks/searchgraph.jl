using SimilaritySearch, Random, JSON


generate_dataset(dim, n) = [rand(Float32, dim) for i in 1:n]

function main_search_graph(perf, S, k; optimize_parameters, parallel=false, opts...)
    println("==============  SearchGraph ================")
    println("=== objects: $(length(S)), dim=$(length(S[1])), knn: $k")

    start = time()
    G = SearchGraph(; opts...)
    !optimize_parameters && delete!(G.callbacks, :parameters)
    #G.neighborhood.logbase = 2
    #G.neighborhood.minsize = 1
    G.neighborhood.logbase = Inf
    G.neighborhood.minsize = 10
    G.neighborhood.reduce = SatNeighborhood()

    append!(G, S; parallel, parallel_firstblock=30_000, parallel_block=10_000)
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
                parallel=false,
                verbose=false
            )
        end
    end
end

main()
