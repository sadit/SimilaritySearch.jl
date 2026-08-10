using Documenter, SimilaritySearch

makedocs(;
    modules=[SimilaritySearch],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/SimilaritySearch.jl/blob/{commit}{path}#L{line}",
    sitename="SimilaritySearch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/SimilaritySearch.jl",
        assets=String[],
        size_threshold=400_000,
        size_threshold_warn=250_000,
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => [
            "tutorial/index.md",
            "tutorial/databases.md",
            "tutorial/distances.md",
            "tutorial/searchgraph.md",
            "tutorial/operations.md",
            "tutorial/parallelism.md",
            "tutorial/persistence.md",
            "tutorial/logging.md",
            "tutorial/invertedfiles.md",
            "tutorial/quantization_and_bitsketches.md",
        ],
        "API" => "api.md"
    ],
    doctest=false,
    warnonly=true  #Documenter.except(:missing_docs, :missing_docs)
)

deploydocs(;
    repo="github.com/sadit/SimilaritySearch.jl",
    devbranch="main",
    devurl="dev",
    versions=["stable" => "v^", "v#.#", "dev" => "dev"],
    push_preview=true,
)
