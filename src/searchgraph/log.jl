# This file is a part of SimilaritySearch.jl

function LOG(log::InformativeLog, event::Symbol, index::SearchGraph, ctx::SearchGraphContext, sp::Int, ep::Int)
    timed_log_fun(log) do 
        n = length(index)
        mem = ceil(Int, Base.gc_live_bytes() / 2^30)
        maxrss = ceil(Int, Sys.maxrss() / 2^30)
        println(stderr, log.prompt, "$event sp=$sp ep=$ep n=$n $(index.algo[]) mem=$(mem)GB max-rss=$(maxrss)GB $(Dates.now())")

        if event === :add_vertex!
            x = quantile(length.(index.adj.end_point[sp:ep]), 0:0.25:1.0)
            println(stderr, "LOG n.size quantiles:", x)
        end
    end
end

#=
struct StorageLog<IOType>
    neighborsfile::String
    databasefile::String
    nfile::IOType
    dfile::IOType
end

function LOG(log::IncrementalStorageLog)
end
=#