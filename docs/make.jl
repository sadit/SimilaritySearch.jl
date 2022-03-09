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
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/sadit/SimilaritySearch.jl",
    devbranch="main",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#"]
)
