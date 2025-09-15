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
    last::Ref{Float64}
    lock::Threads.SpinLock

    InformativeLog(dt::Float64=1.0) = new(dt, Ref(0.0), Threads.SpinLock())
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
        println(stderr, "LOG $event $(typeof(index)) sp=$sp ep=$ep n=$n $(Dates.now())")
    end
end
