# This file is part of SimilaritySearch.jl

using Dates
export AbstractLog, InformativeLog, LOG

abstract type AbstractLog end

struct LogList <: AbstractLog
    list::Vector{AbstractLog}

    LogList() = new(AbstractLog[InformativeLog()])
    LogList(v) = new(v)
end

function LOG(log::LogList, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Int, ep::Int)
    for log in log.list
        LOG(log, event, index, ctx, sp, ep)
    end
end

"""

Informative log. It generates an output each some seconds
"""
struct InformativeLog <: AbstractLog
    dt::Float64
    prompt::String
    last::Ref{Float64}
    lock::Threads.SpinLock

    InformativeLog(dt::Float64=1.0; prompt="LOG") = new(dt, prompt, Ref(0.0), Threads.SpinLock())
end

function timed_log_fun(fun::Function, log::InformativeLog)
    if trylock(log.lock)
        now = time()
        if log.last[] + log.dt < now
            fun()
            log.last[] = now
        end

        unlock(log.lock)
    end
end

function LOG(log::InformativeLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Int, ep::Int)
    timed_log_fun(log) do 
        n = length(index)
        mem = ceil(Int, Sys.total_memory() / 2^20)
        maxrss = ceil(Int, Sys.maxrss() / 2^20)
        println(stderr, log.prompt, " $event $(typeof(index)) sp=$sp ep=$ep n=$n mem=$(mem) max-rss=$(maxrss) $(Dates.now())")
    end
end
