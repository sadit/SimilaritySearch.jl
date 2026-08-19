# This file is a part of SimilaritySearch.jl

# Exercises the LOG event contract documented on `AbstractLog` (src/log.jl): every mutation must
# emit exactly one `:add!` event per logical batch (no duplicate/nested double-logging when one
# mutating function delegates to another), covering exactly the ids that were actually added, and
# a genuine no-op (e.g. ExhaustiveSearch's `index!`) must emit `:info`, never `:add!`.
using SimilaritySearch, SimilaritySearch.InvertedFiles
import SimilaritySearch: ParallelExhaustiveSearch
using Test, Random
Random.seed!(0)

struct RecorderLog <: AbstractLog
    events::Vector{Tuple{Symbol,Int,Int}}
end
RecorderLog() = RecorderLog(Tuple{Symbol,Int,Int}[])

function SimilaritySearch.LOG(log::RecorderLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)
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

@testset "LOG events: exactly-once :add!, no duplicate/no-op misnaming" begin
    dim, n = 4, 50

    @testset "SearchGraph" begin
        db = MatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        graph = SearchGraph(Dist.L2(), db)
        ctx = SearchGraphContext(logger=rec)
        index!(graph, ctx)
        @test all(e -> e[1] === :add!, rec.events)
        @test covers_exactly(rec.events, n)
    end

    @testset "InvertedFile" begin
        db = VectorDatabase([sort!(unique(rand(1:64, 8))) for _ in 1:n])
        rec = RecorderLog()
        idx = InvertedFile(64, Dist.Sets.Jaccard())
        ctx = InvertedFileContext(logger=rec)
        append_items!(idx, ctx, db)
        # append_items! delegates entirely to a single index! call, so this must be exactly one event
        @test length(rec.events) == 1
        @test rec.events[1] == (:add!, 1, n)
    end

    @testset "DictInvertedFile" begin
        db = VectorDatabase([Set(rand(1:64, 5)) for _ in 1:n])
        rec = RecorderLog()
        idx = DictInvertedFile(Int, Dist.Sets.Jaccard())
        ctx = InvertedFileContext(logger=rec)
        append_items!(idx, ctx, db)
        @test length(rec.events) == 1
        @test rec.events[1] == (:add!, 1, n)
    end

    @testset "ExhaustiveSearch" begin
        db = BlockMatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        seq = ExhaustiveSearch(Dist.L2(), db)
        ctx = GenericContext(logger=rec)
        # ExhaustiveSearch's initial `db` is never logged (being in `db` trivially *is* being
        # indexed for a brute-force index, with no catch-up step) -- only the appended batch is.
        append_items!(seq, ctx, MatrixDatabase(rand(Float32, dim, 5)))
        @test rec.events == [(:add!, n + 1, n + 5)]

        empty!(rec.events)
        index!(seq, ctx)  # a true no-op: must be :info, never :add!
        @test length(rec.events) == 1
        @test rec.events[1][1] === :info
    end

    @testset "ParallelExhaustiveSearch" begin
        db = BlockMatrixDatabase(rand(Float32, dim, n))
        rec = RecorderLog()
        pex = ParallelExhaustiveSearch(Dist.L2(), db)
        ctx = GenericContext(logger=rec)
        append_items!(pex, ctx, MatrixDatabase(rand(Float32, dim, 5)))
        @test rec.events == [(:add!, n + 1, n + 5)]

        empty!(rec.events)
        index!(pex, ctx)  # a true no-op: must be :info, never :add!
        @test length(rec.events) == 1
        @test rec.events[1][1] === :info
    end

    @testset "SpatialAccessTree/Sat" begin
        db = MatrixDatabase(rand(Float32, dim, n))
        ctx = GenericContext()

        rec = RecorderLog()
        sat = Sat(db; dist=Dist.L2())
        index!(sat, GenericContext(logger=rec), SatInitialPartition())
        @test rec.events == [(:add!, 1, n)]

        # nparts == 1 delegates internally to the SatInitialPartition method -- must not double-log
        rec2 = RecorderLog()
        sat2 = Sat(db; dist=Dist.L2())
        index!(sat2, GenericContext(logger=rec2), RandomInitialPartition(nparts=1))
        @test rec2.events == [(:add!, 1, n)]

        rec3 = RecorderLog()
        sat3 = Sat(db; dist=Dist.L2())
        index!(sat3, GenericContext(logger=rec3), RandomInitialPartition(nparts=8))
        @test rec3.events == [(:add!, 1, n)]
    end
end
