# Means over repetitions of results/threshold.tsv, one table per build mode.
using DelimitedFiles, Statistics, Printf
data, header = readdlm(joinpath(@__DIR__, "results", "threshold.tsv"), '\t'; header=true)
col(name) = findfirst(==(name), vec(header))
for mode in ("seq", "par")
    println("\n## build: ", mode)
    println("| n | T | build s | edges | recall tuned | evals tuned | recall fixed | evals fixed |")
    println("|---|---|---|---|---|---|---|---|")
    for n in unique(data[:, col("n")]), T in unique(data[:, col("T")])
        rows = findall(i -> data[i, col("n")] == n && data[i, col("T")] == T && data[i, col("mode")] == mode, axes(data, 1))
        m(name) = mean(Float64.(data[rows, col(name)]))
        @printf("| %d | %d | %.2f | %.0f | %.3f | %.0f | %.3f | %.0f |\n", n, T, m("build_s"), m("edges"), m("recall_tuned"), m("evals_tuned"), m("recall_fixed"), m("evals_fixed"))
    end
end
