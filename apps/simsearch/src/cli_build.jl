function parse_build_args(args::Vector{String})
    s = ArgParseSettings(prog="simsearch build",
        description="Build a similarity search index over a dataset and save it to disk.")
    @add_arg_table! s begin
        "--type"
            help = "index type: ExhaustiveSearch, ParallelExhaustiveSearch, SearchGraph"
            required = true
        "--dataset"
            help = "dataset spec 'path.h5:key' or 'path.jld2:key' (columns = object vectors)"
            required = true
        "--distance"
            help = "distance name, e.g. Dist.SqL2 or SqL2 (see README for the full list)"
            required = true
        "--save"
            help = "output index path, must end in .jld2 (the dataset is never embedded)"
            required = true
        "--minrecall"
            help = "target minimum recall for SearchGraph autotuning (optimize_index! with MinRecall); ignored for exact indexes"
            arg_type = Float64
        "--logbase"
            help = "SearchGraph neighborhood growth log-base (Neighborhood.logbase); ignored for exact indexes"
            arg_type = Float64
            default = 1.3
        "--logbase-callback"
            help = "SearchGraph periodic-callback log-base (SearchGraphContext.logbase_callback); ignored for exact indexes"
            arg_type = Float64
            default = 1.5
        "--hints-logbase"
            help = "SearchGraph entry-point hints log-base (RandomHints.logbase); ignored for exact indexes"
            arg_type = Float64
            default = 1.1
    end
    parse_args(args, s)
end

function cmd_build(args::Vector{String})
    o = parse_build_args(args)
    path, key = split_pathkey(o["dataset"])
    X = MatrixDatabase(read_matrix(path, key))
    dist = parse_distance(o["distance"])
    index = build_index(o["type"], dist, X)

    if index isa SearchGraph
        ctx = SearchGraphContext(;
            neighborhood=Neighborhood(; logbase=Float32(o["logbase"]), filter=SatNeighborhood()),
            logbase_callback=Float32(o["logbase-callback"]),
            hints_callback=RandomHints(; logbase=Float32(o["hints-logbase"])))
        index!(index, ctx)
        if o["minrecall"] !== nothing
            optimize_index!(index, ctx, MinRecall(Float32(o["minrecall"])))
        end
    else
        ctx = GenericContext()
        index!(index, ctx)
        if o["minrecall"] !== nothing
            @warn "--minrecall only applies to --type SearchGraph; ignored for $(o["type"])"
        end
    end

    save_index(o["save"], index)
    println("saved $(o["type"]) index with $(length(index)) objects to $(o["save"])")
end
