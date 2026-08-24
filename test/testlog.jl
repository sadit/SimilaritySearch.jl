# This file is a part of SimilaritySearch.jl

# Exercises the two logging channels (src/log.jl).
#
# `OBSERVE` carries the strict one: every mutation must emit exactly one `:add!` event per logical
# batch (no duplicate/nested double-logging when one mutating function delegates to another),
# covering exactly the ids that were actually added, and a genuine no-op (e.g. ExhaustiveSearch's
# `index!`) must emit no event at all -- it is a message, not an event.
#
# `INFORM` carries the loose one, and what is tested about it is the property the split exists for:
# emptying `ctx.reporters` silences the library completely, without disturbing observation, and it
# reaches into the contexts the library builds for itself.
using SimilaritySearch, SimilaritySearch.InvertedFiles
import SimilaritySearch: ParallelExhaustiveSearch, AbstractObserver, AbstractReporter, OBSERVE
using Test, Random, Logging
Random.seed!(0)

struct RecorderLog <: AbstractObserver
    events::Vector{Tuple{Symbol,Int,Int}}
end
RecorderLog() = RecorderLog(Tuple{Symbol,Int,Int}[])

function SimilaritySearch.OBSERVE(log::RecorderLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)
    push!(log.events, (event, Int(sp), Int(ep)))
end

"""
    covers_exactly(events, n)

`true` iff `events`' `(sp, ep)` ranges, taken together, tile `1:n` with no gaps and no overlaps --
i.e. every added object was logged exactly once, by exactly one event.
"""
function covers_exactly(events, n)
    ranges = sort([(sp, ep) for (_, sp, ep) in events])
    isempty(ranges) && return n == 0
    ranges[1][1] == 1 || return false
    ranges[end][2] == n || return false
    for i in 2:length(ranges)
        ranges[i][1] == ranges[i-1][2] + 1 || return false
    end
    true
end

"""
    capture_stderr(f) -> String

Everything `f` writes to `stderr`. Used to assert that a silenced context writes *nothing at all*,
which is a stronger claim than "the reporter did not fire" -- it also catches a `println(stderr,
...)` that never entered the channel.
"""
function capture_stderr(f)
    old = stderr
    rd, wr = redirect_stderr()
    try
        f()
    finally
        redirect_stderr(old)
        close(wr)
    end

    read(rd, String)
end

@testset "OBSERVE events: exactly-once :add!, no duplicate/no-op misnaming" begin
    dim, n = 4, 50

    @testset "SearchGraph" begin
        db = MatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        graph = SearchGraph(Dist.L2(), db)
        ctx = SearchGraphContext(reporters=[], observers=rec)
        index!(graph, ctx)
        @test all(e -> e[1] === :add!, rec.events)
        @test covers_exactly(rec.events, n)
    end

    @testset "InvertedFile" begin
        db = VectorDatabase([sort!(unique(rand(1:64, 8))) for _ in 1:n])
        rec = RecorderLog()
        idx = InvertedFile(64, Dist.Sets.Jaccard())
        ctx = InvertedFileContext(reporters=[], observers=rec)
        append_items!(idx, ctx, db)
        # append_items! delegates entirely to a single index! call, so this must be exactly one event
        @test length(rec.events) == 1
        @test rec.events[1] == (:add!, 1, n)
    end

    @testset "DictInvertedFile" begin
        db = VectorDatabase([Set(rand(1:64, 5)) for _ in 1:n])
        rec = RecorderLog()
        idx = DictInvertedFile(Int, Dist.Sets.Jaccard())
        ctx = InvertedFileContext(reporters=[], observers=rec)
        append_items!(idx, ctx, db)
        @test length(rec.events) == 1
        @test rec.events[1] == (:add!, 1, n)
    end

    @testset "ExhaustiveSearch" begin
        db = BlockMatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        seq = ExhaustiveSearch(Dist.L2(), db)
        ctx = GenericContext(reporters=[], observers=rec)
        # ExhaustiveSearch's initial `db` is never logged (being in `db` trivially *is* being
        # indexed for a brute-force index, with no catch-up step) -- only the appended batch is.
        append_items!(seq, ctx, MatrixDatabase(rand(Float32, dim, 5)))
        @test rec.events == [(:add!, n + 1, n + 5)]

        empty!(rec.events)
        index!(seq, ctx)  # a true no-op: nothing structural happened, so no event at all
        @test isempty(rec.events)
    end

    @testset "ParallelExhaustiveSearch" begin
        db = BlockMatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        pex = ParallelExhaustiveSearch(Dist.L2(), db)
        ctx = GenericContext(reporters=[], observers=rec)
        append_items!(pex, ctx, MatrixDatabase(rand(Float32, dim, 5)))
        @test rec.events == [(:add!, n + 1, n + 5)]

        empty!(rec.events)
        index!(pex, ctx)  # a true no-op: nothing structural happened, so no event at all
        @test isempty(rec.events)
    end

    @testset "SpatialAccessTree/Sat" begin
        db = MatrixDatabase(rand(Float32, dim, n))

        rec = RecorderLog()
        sat = Sat(db; dist=Dist.L2())
        index!(sat, GenericContext(reporters=[], observers=rec), SatInitialPartition())
        @test rec.events == [(:add!, 1, n)]

        # nparts == 1 delegates internally to the SatInitialPartition method -- must not double-log
        rec2 = RecorderLog()
        sat2 = Sat(db; dist=Dist.L2())
        index!(sat2, GenericContext(reporters=[], observers=rec2), RandomInitialPartition(nparts=1))
        @test rec2.events == [(:add!, 1, n)]

        rec3 = RecorderLog()
        sat3 = Sat(db; dist=Dist.L2())
        index!(sat3, GenericContext(reporters=[], observers=rec3), RandomInitialPartition(nparts=8))
        @test rec3.events == [(:add!, 1, n)]
    end
end

@testset "INFORM: silencing, throttling, and what a context inherits" begin
    dim, n = 4, 50

    @testset "reporters=[] silences everything, observers keep working" begin
        db = MatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        graph = SearchGraph(Dist.L2(), db)
        # verbose=true asks for *every* message the library can produce, including the
        # per-configuration optimization detail; with no reporter, none of it reaches stderr.
        ctx = SearchGraphContext(verbose=true, reporters=[], observers=rec)
        captured = with_logger(NullLogger()) do          # @warn is a different channel, by design
            capture_stderr() do
                index!(graph, ctx)
            end
        end

        @test isempty(captured)
        @test covers_exactly(rec.events, n)              # silence did not disturb observation
    end

    @testset "the same run does report when given a reporter" begin
        db = MatrixDatabase(rand(Float32, dim, n))
        buf = IOBuffer()
        graph = SearchGraph(Dist.L2(), db)
        ctx = SearchGraphContext(verbose=true, reporters=InformativeLog(buf; dt=0))
        index!(graph, ctx)
        s = String(take!(buf))
        @test occursin("add!", s)
        @test occursin("n.size-quantiles", s)            # the SearchGraph-specific detail survives
    end

    @testset "dt <= 0 drops nothing" begin
        buf = IOBuffer()
        ctx = GenericContext(; reporters=InformativeLog(buf; dt=0))
        seq = ExhaustiveSearch(Dist.L2(), VectorDatabase(Vector{Float32}[]))
        for _ in 1:20
            push_item!(seq, ctx, rand(Float32, dim))
        end

        @test count(==('\n'), String(take!(buf))) == 20
    end

    @testset "dt > 0 throttles" begin
        buf = IOBuffer()
        ctx = GenericContext(; reporters=InformativeLog(buf; dt=1000))
        seq = ExhaustiveSearch(Dist.L2(), VectorDatabase(Vector{Float32}[]))
        for _ in 1:20
            push_item!(seq, ctx, rand(Float32, dim))
        end

        @test count(==('\n'), String(take!(buf))) == 1
    end

    @testset "an internally built context inherits reporters, never observers" begin
        # EpsilonHints builds its own GenericContext for a scratch ExhaustiveSearch and runs
        # `neardup` on it. Its progress must reach the caller's reporters (otherwise silencing
        # cannot be honored there) while its `:add!` events -- which are about the scratch index's
        # ids, not this graph's -- must not reach the caller's observers.
        m = 512
        db = MatrixDatabase(rand(Float32, dim, m))
        buf = IOBuffer()
        rec = RecorderLog()
        graph = SearchGraph(Dist.L2(), db)
        ctx = SearchGraphContext(verbose=true, reporters=InformativeLog(buf; dt=0), observers=rec)
        index!(graph, ctx)
        take!(buf)

        # invoked directly rather than through the callback schedule, so the test does not depend
        # on how often `execute_callbacks!` decides to run this one
        SimilaritySearch.execute_callback!(graph, ctx, EpsilonHints(; quantile=0.05))

        @test occursin("neardup>", String(take!(buf)))   # reporters travel
        @test covers_exactly(rec.events, m)              # observers do not: no foreign ranges
    end
end
