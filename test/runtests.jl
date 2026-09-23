# This file is a part of SimilaritySearch.jl
using SimilaritySearch, LinearAlgebra
#using JET

# Every file, always: there is no reduced mode. What used to be `FAST_TESTS` shrank dataset
# sizes, which is not where the time goes -- in CI, of a 25.7 min run, 5.5 min was spent
# inside testsets and 18.9 min compiling between them, so shrinking the data could not have
# moved the number much. To iterate quickly, run *fewer files* instead:
#
#     julia --project=. -e 'using Pkg; Pkg.test(test_args=["searchgraph"])'
#     julia -t auto --project=. test/runtests.jl searchgraph quantsketch
#
# Each argument is matched as a substring against the file names below (case-insensitive), so
# `searchgraph` runs testsearchgraph.jl and `spatialaccess` runs both SAT files. A pattern that
# matches nothing is an error listing the available names, rather than a silent empty run.
# Faster still, for a tight edit/run loop, is one persistent session with Revise:
# `using Revise, SimilaritySearch, Test; includet("test/testsearchgraph.jl")`.
const TESTFILES = [
    "testbatches.jl",
    "testdistances.jl",
    "testdb.jl",
    "testmmapdb.jl",
    "testlog.jl",
    "testresults.jl",
    "testsparse.jl",
    "testscalarquant.jl",
    "testspherical.jl",
    "testexactseq.jl",
    "testexact.jl",
    "testparallelexhaustive.jl",
    "testhsp.jl",
    "testselection.jl",
    "testadj.jl",
    "testsearchgraph.jl",
    "testallknn.jl",
    "testclosestpair.jl",
    "testindexingprefixes.jl",
    "testintersections.jl",
    "testinvertedfiles.jl",
    "testprojections.jl",
    "testquantsketch.jl",
    "testspatialaccesstree.jl",
    "testspatialaccesstreeopt.jl",
    "testbktree.jl",
]

selected(f) = isempty(ARGS) || any(a -> occursin(lowercase(a), lowercase(f)), ARGS)
const SELECTED = filter(selected, TESTFILES)
isempty(SELECTED) && error("no test file matches $(ARGS); available: " * join(TESTFILES, ", "))

# Aqua's checks are about the package as a whole, not about any one file, so they belong to a
# complete run. They are pinned to 1.12 because its findings -- ambiguities above all -- differ
# between Julia versions, and 1.12 is what CI runs. (Written as `>=`: the old `VERSION ==
# v"1.10"` never fired, since VERSION is 1.10.12 while v"1.10" means v"1.10.0".)
if VERSION >= v"1.12" && isempty(ARGS)
    using Aqua
    Aqua.test_all(SimilaritySearch, ambiguities=false)
    Aqua.test_ambiguities([SimilaritySearch])
end

function create_sequence(dim, sort, range=1:10)
    s = rand(range, dim)
    if sort
        sort!(s)
        s = unique(s)
    end

    s
end

length(SELECTED) < length(TESTFILES) &&
    @info "running $(length(SELECTED)) of $(length(TESTFILES)) test files" SELECTED

for f in SELECTED
    include(f)
end
