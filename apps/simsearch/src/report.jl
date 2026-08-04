"""
    descriptive_stats(v) -> NamedTuple

Basic descriptive statistics (`n`, `mean`, `std`, `min`, `median`, `max`) for a numeric vector.
"""
function descriptive_stats(v::AbstractVector{<:Real})
    isempty(v) && return (n=0, mean=NaN, std=NaN, min=NaN, median=NaN, max=NaN)
    q = quantile(v, (0.0, 0.5, 1.0))
    (n=length(v), mean=mean(v), std=std(v), min=q[1], median=q[2], max=q[3])
end

"""
    fmtval(x)

Formats a value for report display, trimming the trailing `.0` that Julia's array-literal type
promotion otherwise leaves on integer-valued statistics mixed into a `Float64` column.
"""
fmtval(x::AbstractFloat) = isnan(x) ? "NaN" : (isinteger(x) ? string(Int(x)) : string(round(x, digits=4)))
fmtval(x) = string(x)

function stats_table_html(rows::Vector{<:Pair})
    io = IOBuffer()
    println(io, "<table>")
    for (label, value) in rows
        println(io, "<tr><th>$(label)</th><td>$(fmtval(value))</td></tr>")
    end
    println(io, "</table>")
    String(take!(io))
end

function stats_table_text(rows::Vector{<:Pair})
    io = IOBuffer()
    for (label, value) in rows
        println(io, @sprintf("%-28s %s", label, fmtval(value)))
    end
    String(take!(io))
end

"""
    svg_histogram(values; nbins=30, width=480, height=160, title="") -> String

Renders a static inline-SVG bar-chart histogram of `values` as a self-contained `<svg>` string.
"""
function svg_histogram(values::AbstractVector{<:Real}; nbins::Int=30, width::Int=480,
                        height::Int=160, title::AbstractString="")
    if isempty(values)
        return "<p><em>$(title): no data</em></p>"
    end

    h = fit(Histogram, collect(Float64, values); nbins=nbins)
    edges = collect(h.edges[1])
    weights = h.weights
    nb = length(weights)
    maxw = maximum(weights)
    maxw = maxw == 0 ? 1 : maxw

    margin_left, margin_bottom, margin_top = 40, 24, 20
    plot_w = width - margin_left - 8
    plot_h = height - margin_bottom - margin_top
    barw = plot_w / nb

    io = IOBuffer()
    println(io, "<svg viewBox=\"0 0 $width $height\" width=\"$width\" height=\"$height\" ",
                "xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\"$(title) histogram\">")
    println(io, "<text x=\"$(width÷2)\" y=\"14\" text-anchor=\"middle\" font-size=\"12\" ",
                "font-family=\"sans-serif\">$(title)</text>")
    for i in 1:nb
        w = weights[i]
        barh = plot_h * (w / maxw)
        x = margin_left + (i - 1) * barw
        y = margin_top + (plot_h - barh)
        println(io, "<rect x=\"$(round(x, digits=2))\" y=\"$(round(y, digits=2))\" ",
                    "width=\"$(round(barw - 1, digits=2))\" height=\"$(round(barh, digits=2))\" ",
                    "fill=\"#4472c4\"><title>[$(round(edges[i],digits=3)), $(round(edges[i+1],digits=3))): $w</title></rect>")
    end
    println(io, "<line x1=\"$margin_left\" y1=\"$(margin_top+plot_h)\" x2=\"$(width-8)\" ",
                "y2=\"$(margin_top+plot_h)\" stroke=\"#333\" stroke-width=\"1\"/>")
    println(io, "<text x=\"$margin_left\" y=\"$(height-4)\" font-size=\"10\" ",
                "font-family=\"sans-serif\">$(round(edges[1], digits=3))</text>")
    println(io, "<text x=\"$(width-8)\" y=\"$(height-4)\" text-anchor=\"end\" font-size=\"10\" ",
                "font-family=\"sans-serif\">$(round(edges[end], digits=3))</text>")
    println(io, "</svg>")
    String(take!(io))
end

"""
    html_page(title, body_html) -> String

Wraps `body_html` in a minimal, self-contained HTML document.
"""
function html_page(title::AbstractString, body_html::AbstractString)
    """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>$title</title>
    <style>
    body { font-family: sans-serif; margin: 2em; color: #222; }
    h1 { font-size: 1.4em; }
    h2 { font-size: 1.1em; margin-top: 1.5em; }
    table { border-collapse: collapse; margin: 0.5em 0 1em 0; }
    th, td { padding: 4px 10px; border: 1px solid #ccc; text-align: left; }
    th { background: #f2f2f2; }
    .charts { display: flex; flex-wrap: wrap; gap: 1.5em; }
    </style>
    </head>
    <body>
    <h1>$title</h1>
    $body_html
    </body>
    </html>
    """
end

function write_text_report(path::Union{Nothing,AbstractString}, text::AbstractString)
    print(text)
    path !== nothing && write(path, text)
end

function write_html_report(path::AbstractString, title::AbstractString, body_html::AbstractString)
    write(path, html_page(title, body_html))
end
