# This file is a part of SimilaritySearch.jl
using SimilaritySearch, LinearAlgebra
#using JET

# Fast dev loop: `FAST_TESTS=true julia -t auto --project=. -e 'using Pkg; Pkg.test()'`
# shrinks the handful of tests whose cost actually scales with dataset size/iteration count
# (SearchGraph/InvertedFile construction, optimize_index! autotuning, SpatialAccessTree),
# without skipping any test file. It's meant for quick iteration, NOT as a substitute for
# a full `Pkg.test()` run (unset, the default) before committing/pushing.
@isdefined(FAST_TESTS) || (const FAST_TESTS = get(ENV, "FAST_TESTS", "false") == "true")

# Routine work targets 1.12 (the only version CI runs); `[compat] julia` still claims
# 1.10+, and @BATCHES keeps its VERSION gates for that. Aqua reports version-dependent
# results (ambiguities especially), so it is pinned to the version everything else is
# checked on -- written as `>=` because the old `VERSION == v"1.10"` never fired: VERSION
# is 1.10.12, while v"1.10" means v"1.10.0".
if VERSION >= v"1.12" && !FAST_TESTS
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

include("testbatches.jl")
include("testdistances.jl")
include("testdb.jl")
include("testmmapdb.jl")
include("testlog.jl")
include("testresults.jl")
include("testsparse.jl")
include("testscalarquant.jl")
include("testspherical.jl")
include("testexactseq.jl")
include("testexact.jl")
include("testparallelexhaustive.jl")
include("testhsp.jl")
include("testselection.jl")
include("testadj.jl")
include("testsearchgraph.jl")
include("testallknn.jl")
include("testclosestpair.jl")
include("testindexingprefixes.jl")
include("testintersections.jl")
include("testinvertedfiles.jl")
include("testprojections.jl")
include("testquantsketch.jl")
include("testspatialaccesstree.jl")
include("testspatialaccesstreeopt.jl")
include("testbktree.jl")
