module SimSearch

using ArgParse, HDF5, JLD2, Accessors, Printf
using Statistics: mean, std, quantile
using StatsBase: Histogram, fit
using SimilaritySearch
using SimilaritySearch: Dist, Exact, MinRecall, Neighborhood, SatNeighborhood,
    RandomHints, SearchGraphContext, GenericContext,
    macrorecall, recallscore, neighbors_length, distance, database

include("datasetio.jl")
include("registry.jl")
include("persistence.jl")
include("report.jl")
include("cli_build.jl")
include("cli_search.jl")
include("cli_evaluate.jl")
include("cli_analyze.jl")

const SUBCOMMANDS = Dict(
    "build" => cmd_build,
    "search" => cmd_search,
    "evaluate" => cmd_evaluate,
    "analyze" => cmd_analyze,
)

function print_top_help(io::IO)
    println(io, """
    simsearch -- build, search, evaluate, and analyze SimilaritySearch.jl indexes.

    Usage: simsearch <subcommand> [options]

    Subcommands:
      build       build and save an index from a dataset
      search      load an index and run batch queries
      evaluate    compare a results file against a gold standard
      analyze     inspect a saved index and its dataset

    Run 'simsearch <subcommand> --help' for subcommand-specific options.
    """)
end

"""
    main(args=ARGS) -> Int

Entry point: dispatches to the requested subcommand. Returns a process exit code.
"""
function (@main)(args::Vector{String}=ARGS)
    if isempty(args) || first(args) in ("-h", "--help")
        print_top_help(stdout)
        return 0
    end

    fn = get(SUBCOMMANDS, args[1], nothing)
    if fn === nothing
        println(stderr, "unknown subcommand '$(args[1])'. Use one of: build, search, evaluate, analyze")
        return 1
    end

    fn(args[2:end])
    return 0
end

end # module
