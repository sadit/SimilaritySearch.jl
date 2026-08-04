function parse_analyze_args(args::Vector{String})
    s = ArgParseSettings(prog="simsearch analyze",
        description="Inspect a saved index and its dataset, reporting general and (for SearchGraph) graph statistics.")
    @add_arg_table! s begin
        "index"
            help = "saved index path (.jld2, produced by 'simsearch build')"
            required = true
        "--dataset"
            help = "dataset spec 'path:key' used to reattach the real data to the loaded index"
            required = true
        "--html"
            help = "write a self-contained HTML report to this path"
        "--out"
            help = "also write the plain-text report to this path"
    end
    parse_args(args, s)
end

function cmd_analyze(args::Vector{String})
    o = parse_analyze_args(args)
    index = load_index(o["index"], o["dataset"])
    n = length(index)
    dim = n > 0 ? length(database(index, 1)) : 0

    common = ["type"=>string(typeof(index).name.name), "n objects"=>n, "dimension"=>dim,
              "distance"=>string(typeof(distance(index)))]

    lines = IOBuffer()
    println(lines, "=== simsearch analyze ===")
    print(lines, stats_table_text(common))

    degs = nothing
    if index isa SearchGraph
        degs = neighbors_length.(Ref(index.adj), 1:n)
        dstats = descriptive_stats(degs)
        beam = index.algo[]
        println(lines, "--- SearchGraph ---")
        print(lines, stats_table_text([
            "hints count"=>length(index.hints),
            "degree n"=>dstats.n, "degree mean"=>dstats.mean, "degree std"=>dstats.std,
            "degree min"=>dstats.min, "degree median"=>dstats.median, "degree max"=>dstats.max,
            "beam bsize"=>beam.bsize, "beam Δ"=>beam.Δ, "beam maxvisits"=>beam.maxvisits]))
    end

    write_text_report(o["out"], String(take!(lines)))

    if o["html"] !== nothing
        body = IOBuffer()
        println(body, stats_table_html(common))
        if index isa SearchGraph
            dstats = descriptive_stats(degs)
            beam = index.algo[]
            println(body, "<h2>SearchGraph</h2>")
            println(body, stats_table_html([
                "hints count"=>length(index.hints),
                "degree mean"=>round(dstats.mean,digits=3), "degree std"=>round(dstats.std,digits=3),
                "degree min"=>dstats.min, "degree median"=>dstats.median, "degree max"=>dstats.max,
                "beam bsize"=>beam.bsize, "beam Δ"=>beam.Δ, "beam maxvisits"=>beam.maxvisits]))
            println(body, "<div class=\"charts\">")
            println(body, svg_histogram(degs; title="node degree distribution"))
            println(body, "</div>")
        end
        write_html_report(o["html"], "simsearch analyze report", String(take!(body)))
    end
end
