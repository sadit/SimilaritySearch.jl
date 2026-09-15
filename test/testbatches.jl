# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test

# Every scheduler name `@BATCHES` accepts, on both paths that reach the dispatcher: a
# literal `scheduler=:sym` (validated and specialized at macro-expansion time) and a
# runtime expression (validated once, right before the batches start).
const BATCH_SCHEDULERS = VERSION >= v"1.11" ?
    (:dynamic, :default, :static, :greedy, :sequential) :
    (:dynamic, :default, :static, :sequential)

function fill_runtime_scheduler(sched::Symbol, n::Int)
    acc = zeros(Int, n)
    @BATCHES 10 scheduler=sched for i in 1:n
        acc[i] = i
    end
    acc
end

@testset "@BATCHES schedulers" begin
    @testset "the global default is :dynamic" begin
        # :static is not usable as a global default: Julia refuses to enter a
        # `@threads :static` region while another one is live *anywhere* in the process,
        # so two unrelated indexes would collide (issue #63).
        @test get_batch_scheduler() === :dynamic
    end

    n = 1000
    expected = collect(1:n)

    @testset "scheduler=$sched (runtime expression)" for sched in BATCH_SCHEDULERS
        @test fill_runtime_scheduler(sched, n) == expected
    end

    @testset ":dynamic as a literal, and @batchid()/@nbatches() under it" begin
        acc = zeros(Int, n)
        @BATCHES 10 scheduler=:dynamic for i in 1:n
            acc[i] = i
        end
        @test acc == expected

        ids = Int[]
        @BATCHES 10 scheduler=:dynamic begin
        @BEGIN
            ids = zeros(Int, @nbatches())
        @LOOP for i in 1:n
            nothing
        end
        @ENDBATCH
            ids[@batchid()] = @batchid()
        end
        @test ids == collect(1:length(ids))
    end

    @testset "concurrent regions" begin
        # the reason :dynamic exists as a name of its own and is the default: several
        # indexes served from one process must be able to run @BATCHES concurrently
        tasks = [Threads.@spawn fill_runtime_scheduler(:dynamic, 20_000) for _ in 1:4]
        @test all(r -> r == collect(1:20_000), fetch.(tasks))
    end

    @testset "invalid names are rejected" begin
        @test_throws ArgumentError set_batch_scheduler!(:bogus)
        @test_throws ArgumentError fill_runtime_scheduler(:bogus, 100)
        @test get_batch_scheduler() === :dynamic  # a rejected name leaves the global alone
    end

    @testset "set_batch_scheduler! round-trip" begin
        prev = get_batch_scheduler()
        try
            for sched in BATCH_SCHEDULERS
                set_batch_scheduler!(sched)
                @test get_batch_scheduler() === sched
            end
        finally
            set_batch_scheduler!(prev)
        end
        @test get_batch_scheduler() === prev
    end
end
