function parse_evaluate_args(args::Vector{String})
    s = ArgParseSettings(prog="simsearch evaluate",
        description="Compare a results file against a gold-standard results file.")
    @add_arg_table! s begin
        "--gold"
            help = "gold results spec: 'path' (fixed ids/dists keys) or 'path:key' (external ids matrix)"
            required = true
        "--results"
            help = "results spec, same grammar as --gold"
            required = true
        "-k", "--k"
            help = "number of neighbors to evaluate per query (default: min(k_gold, k_results))"
            arg_type = Int
        "--html"
            help = "write a self-contained HTML report to this path"
        "--out"
            help = "also write the plain-text report to this path"
    end
    parse_args(args, s)
end

function cmd_evaluate(args::Vector{String})
    o = parse_evaluate_args(args)
    goldI, goldD = load_results_spec(o["gold"])
    resI, resD = load_results_spec(o["results"])

    k = o["k"] !== nothing ? o["k"] : min(size(goldI, 1), size(resI, 1))
    n = min(size(goldI, 2), size(resI, 2))
    goldI = goldI[1:k, 1:n]
    resI = resI[1:k, 1:n]

    overall_recall = macrorecall(goldI, resI, k)
    perquery_recall = [recallscore(view(goldI, :, i), view(resI, :, i)) for i in 1:n]
    recall_stats = descriptive_stats(perquery_recall)

    missing_gold = count(==(0), goldI)
    missing_res = count(==(0), resI)

    lines = IOBuffer()
    println(lines, "=== simsearch evaluate ===")
    println(lines, @sprintf("%-28s %s", "queries", n))
    println(lines, @sprintf("%-28s %s", "k", k))
    println(lines, @sprintf("%-28s %.6f", "macrorecall", overall_recall))
    println(lines, "--- per-query recall ---")
    print(lines, stats_table_text(["n"=>recall_stats.n, "mean"=>recall_stats.mean,
        "std"=>recall_stats.std, "min"=>recall_stats.min, "median"=>recall_stats.median,
        "max"=>recall_stats.max]))
    println(lines, "--- coverage (id==0 sentinel, fewer than k neighbors found) ---")
    println(lines, @sprintf("%-28s %s", "gold missing entries", missing_gold))
    println(lines, @sprintf("%-28s %s", "results missing entries", missing_res))

    gold_dist_stats = goldD === nothing ? nothing : descriptive_stats(vec(goldD[1:k, 1:n]))
    res_dist_stats = resD === nothing ? nothing : descriptive_stats(vec(resD[1:k, 1:n]))
    if gold_dist_stats !== nothing
        println(lines, "--- gold distance distribution ---")
        print(lines, stats_table_text(["n"=>gold_dist_stats.n, "mean"=>gold_dist_stats.mean,
            "std"=>gold_dist_stats.std, "min"=>gold_dist_stats.min,
            "median"=>gold_dist_stats.median, "max"=>gold_dist_stats.max]))
    end
    if res_dist_stats !== nothing
        println(lines, "--- results distance distribution ---")
        print(lines, stats_table_text(["n"=>res_dist_stats.n, "mean"=>res_dist_stats.mean,
            "std"=>res_dist_stats.std, "min"=>res_dist_stats.min,
            "median"=>res_dist_stats.median, "max"=>res_dist_stats.max]))
    end

    write_text_report(o["out"], String(take!(lines)))

    if o["html"] !== nothing
        body = IOBuffer()
        println(body, stats_table_html(["queries"=>n, "k"=>k, "macrorecall"=>round(overall_recall, digits=6),
            "gold missing entries"=>missing_gold, "results missing entries"=>missing_res]))
        println(body, "<h2>Per-query recall</h2>")
        println(body, stats_table_html(["n"=>recall_stats.n, "mean"=>round(recall_stats.mean,digits=4),
            "std"=>round(recall_stats.std,digits=4), "min"=>recall_stats.min,
            "median"=>recall_stats.median, "max"=>recall_stats.max]))
        println(body, "<div class=\"charts\">")
        println(body, svg_histogram(perquery_recall; title="per-query recall"))
        if goldD !== nothing || resD !== nothing
            println(body, "</div><h2>Distance distributions</h2><div class=\"charts\">")
            goldD !== nothing && println(body, svg_histogram(vec(goldD[1:k, 1:n]); title="gold distances"))
            resD !== nothing && println(body, svg_histogram(vec(resD[1:k, 1:n]); title="results distances"))
        end
        println(body, "</div>")
        write_html_report(o["html"], "simsearch evaluate report", String(take!(body)))
    end
end
