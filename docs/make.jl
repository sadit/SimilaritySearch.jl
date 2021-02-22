using SimilaritySearch
using Documenter

makedocs(;
    modules=[SimilaritySearch],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/SimilaritySearch.jl/blob/{commit}{path}#L{line}",
    sitename="SimilaritySearch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sadit.github.io/SimilaritySearch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sadit/SimilaritySearch.jl",
    devbranch="main",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#"]
)
