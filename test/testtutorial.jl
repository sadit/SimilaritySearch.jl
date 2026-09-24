# This file is a part of SimilaritySearch.jl
using SimilaritySearch, Test

"""
Tutorial pages whose `julia` blocks are *executed* by this file, and which therefore must
carry a `# SimilaritySearch vMAJOR.MINOR` marker on their first line.

Two kinds of drift are being guarded against, and they need different guards. The marker says
which version an example was written for -- SimilaritySearch v1.0.0's own README documented a
pre-1.0 API, and a reader had no way to tell. Running the block says the example still works:
when this file was added, `quantization_and_bitsketches.md` had been shipping
`ExhaustiveSearch(Dist.SqL2(), db_sq)` over a quantized database, which raises a MethodError,
because `doctest=false` and nothing ever ran it.

A page can only be listed here if its blocks are **self-contained**: much of this series is
narrative and reuses variables across pages (the Dice example in
`quantization_and_bitsketches.md` says outright that it continues the quickstart's dataset),
and those cannot be executed in isolation. Add a page when it is written or revised; that is
cheaper than discovering the breakage from a user, which is how the last one was found.
"""
const VERSIONED_TUTORIALS = [
    "multibit_sketches.md",
    "sketchedsearch.md",
]

function julia_blocks(path)
    blocks = String[]
    current = nothing
    for line in eachline(path)
        if current === nothing
            startswith(line, "```julia") && (current = String[])
        elseif startswith(line, "```")
            push!(blocks, join(current, "\n"))
            current = nothing
        else
            push!(current, line)
        end
    end
    blocks
end

@testset "tutorial examples run, and say which version they target" begin
    # read the version without TOML: it is a stdlib, but Pkg.test's sandbox only provides
    # what Project.toml's test target declares, and this is not worth a dependency
    pv = match(r"^version\s*=\s*\"([^\"]+)\""m, read(joinpath(@__DIR__, "..", "Project.toml"), String))
    v = VersionNumber(pv[1])
    marker = "# SimilaritySearch v$(v.major).$(v.minor)"
    docs = joinpath(@__DIR__, "..", "docs", "src", "tutorial")

    for page in VERSIONED_TUTORIALS
        path = joinpath(docs, page)
        @test isfile(path)
        blocks = julia_blocks(path)
        @test !isempty(blocks)

        for (i, code) in enumerate(blocks)
            first_line = strip(first(split(code, "\n")))
            @test first_line == marker      # names the version it was written for
            # and still runs: each block in its own module, as a reader would paste it
            mod = Module(Symbol("TutorialBlock_", replace(page, "." => "_"), "_", i))
            Core.eval(mod, :(using SimilaritySearch, Test))
            @test (include_string(mod, code); true)
        end
    end
end
