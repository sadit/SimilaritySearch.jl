function parse_search_args(args::Vector{String})
    s = ArgParseSettings(prog="simsearch search",
        description="Load a saved index and run batch queries against it.")
    @add_arg_table! s begin
        "index"
            help = "saved index path (.jld2, produced by 'simsearch build')"
            required = true
        "--queries"
            help = "queries spec 'path.h5:key' or 'path.jld2:key' (columns = query vectors)"
            required = true
        "--dataset"
            help = "dataset spec 'path:key' used to reattach the real data to the loaded index"
            required = true
        "--results"
            help = "output results path, .h5 or .jld2"
            required = true
        "-k", "--k"
            help = "number of neighbors to retrieve per query"
            arg_type = Int
            default = 10
    end
    parse_args(args, s)
end

function cmd_search(args::Vector{String})
    o = parse_search_args(args)
    index = load_index(o["index"], o["dataset"])
    ctx = index isa SearchGraph ? SearchGraphContext() : GenericContext()
    qpath, qkey = split_pathkey(o["queries"])
    Q = MatrixDatabase(read_matrix(qpath, qkey))
    ids, dists = searchbatch(index, ctx, Q, o["k"])
    write_results(o["results"], Matrix{Int32}(ids), dists)
    println("searched $(length(Q)) queries (k=$(o["k"])) -> $(o["results"])")
end
