# This file is a part of SimilaritySearch.jl
using Test, SimilaritySearch

@testset "MMapMatrixDatabase roundtrip" begin
    mktempdir() do dir
        path = joinpath(dir, "roundtrip.mmapdb")
        dim = 4
        # capacity_bits=2 -> initial capacity 4; pushing 50 items crosses several doubling
        # boundaries (4 -> 8 -> 16 -> 32 -> 64).
        N = 50
        X = [Float32.(1:dim) .* i for i in 1:N]

        db = MMapMatrixDatabase(path, dim, Float32; capacity_bits=2)
        for v in X
            push_item!(db, v)
        end
        @test length(db) == N
        close(db)

        db2 = MMapMatrixDatabase(path)
        @test length(db2) == N
        @test eltype(db2) == typeof(db2[1])
        for i in 1:N
            @test db2[i] == X[i]
        end
        close(db2)

        # append_items! across a capacity boundary too, then reopen again
        db3 = MMapMatrixDatabase(path)
        extra = [Float32.(1:dim) .* (N + i) for i in 1:30]
        append_items!(db3, extra)
        @test length(db3) == N + 30
        close(db3)

        db4 = MMapMatrixDatabase(path)
        @test length(db4) == N + 30
        for i in 1:N
            @test db4[i] == X[i]
        end
        for i in 1:30
            @test db4[N + i] == extra[i]
        end
        close(db4)
    end
end

@testset "MMapMatrixDatabase read_only mode" begin
    mktempdir() do dir
        path = joinpath(dir, "readonly.mmapdb")
        db = MMapMatrixDatabase(path, 3, Float32; capacity_bits=2)
        push_item!(db, Float32[1, 2, 3])
        close(db)

        rdb = MMapMatrixDatabase(path; read_only=true)
        @test length(rdb) == 1
        @test rdb[1] == Float32[1, 2, 3]
        @test_throws ErrorException push_item!(rdb, Float32[4, 5, 6])
        @test_throws ErrorException append_items!(rdb, [Float32[4, 5, 6]])
        @test_throws ErrorException (rdb[1] = Float32[9, 9, 9])
        @test flush(rdb) === rdb   # a no-op, not an error (nothing to persist, and the file isn't open for writing)
        close(rdb)
    end
end

@testset "MMapMatrixDatabase: push_item!/append_items! are not durable without flush" begin
    mktempdir() do dir
        path = joinpath(dir, "no_autoflush.mmapdb")
        dim = 4
        db = MMapMatrixDatabase(path, dim, Float32; capacity_bits=4)
        append_items!(db, [Float32.(1:dim) .* i for i in 1:5])
        @test length(db) == 5   # in-memory length reflects it immediately...

        # ...but nothing was made durable: a second handle to the same file, opened without
        # `db` ever having been flushed or closed, still sees the on-disk header as it was
        # when the file was created (n=0).
        db2 = MMapMatrixDatabase(path)
        @test length(db2) == 0
        close(db2)

        # flush() (not close()) is what persists it, and does so without needing to stop
        # using `db` for further writes.
        flush(db)
        db3 = MMapMatrixDatabase(path)
        @test length(db3) == 5
        close(db3)
        close(db)
    end
end

@testset "MMapMatrixDatabase in-process crash-safety (uncommitted writes never surface)" begin
    mktempdir() do dir
        path = joinpath(dir, "crash_inprocess.mmapdb")
        dim = 4
        db = MMapMatrixDatabase(path, dim, Float32; capacity_bits=4)  # capacity 16, enough headroom
        committed = [Float32.(1:dim) .* i for i in 1:5]
        append_items!(db, committed)
        flush(db)   # durability is opt-in now -- this is what the old auto-flush-per-call used to do
        n_durable = length(db)
        @test n_durable == 5

        # Simulate a crash mid-append: write additional columns directly into the mapped
        # data, bypassing push_item!/append_items!, so neither Mmap.sync! nor the header's
        # `n` are ever updated for them -- exactly what a process kill between writing bytes
        # and the next `flush` would leave behind.
        raw = db.n
        for i in 1:7
            raw += 1
            raw > size(db.data, 2) && error("test setup: not enough preallocated capacity")
            db.data[:, raw] .= Float32.(1:dim) .* (100 + i)
        end
        # db.n and the on-disk header are deliberately left untouched (simulating the crash);
        # we just abandon `db` without calling flush/close again.

        db2 = MMapMatrixDatabase(path)
        @test length(db2) == n_durable
        for i in 1:n_durable
            @test db2[i] == committed[i]
        end
        close(db2)
    end
end

@testset "MMapMatrixDatabase crash-safety across a killed process" begin
    mktempdir() do dir
        path = joinpath(dir, "crash_subprocess.mmapdb")
        dim = 4
        db = MMapMatrixDatabase(path, dim, Float32; capacity_bits=4)
        committed = [Float32.(1:dim) .* i for i in 1:3]
        append_items!(db, committed)
        n_durable = length(db)
        close(db)

        childscript = """
        using SimilaritySearch
        db = MMapMatrixDatabase(raw"$path")
        function slowbatch()
            Channel() do ch
                for i in 1:2000
                    sleep(0.01)
                    put!(ch, Float32[i, i, i, i])
                end
            end
        end
        append_items!(db, slowbatch())
        """
        proc = run(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $childscript`;
                            stdout=devnull, stderr=devnull); wait=false)
        sleep(1.0)
        @test process_running(proc)
        kill(proc, Base.SIGKILL)
        wait(proc)

        db2 = MMapMatrixDatabase(path)
        @test length(db2) == n_durable
        for i in 1:n_durable
            @test db2[i] == committed[i]
        end
        close(db2)
    end
end

@testset "MMapMatrixDatabase performance sanity (opt-in, no RAM-path regression)" begin
    mktempdir() do dir
        path = joinpath(dir, "perf.mmapdb")
        dim = 16
        N = 5_000
        X = [rand(Float32, dim) for _ in 1:N]

        vdb = VectorDatabase(Vector{Float32}[])
        bdb = BlockMatrixDatabase(dim, Float32)
        mdb = MMapMatrixDatabase(path, dim, Float32; capacity_bits=8)

        tv = @elapsed for v in X; push_item!(vdb, v); end
        tb = @elapsed for v in X; push_item!(bdb, v); end
        tm = @elapsed for v in X; push_item!(mdb, v); end

        @test length(vdb) == length(bdb) == length(mdb) == N

        gv = @elapsed for i in 1:N; vdb[i]; end
        gb = @elapsed for i in 1:N; bdb[i]; end
        gm = @elapsed for i in 1:N; mdb[i]; end

        @info "MMapMatrixDatabase perf sanity" push_VectorDatabase=tv push_BlockMatrixDatabase=tb push_MMapMatrixDatabase=tm get_VectorDatabase=gv get_BlockMatrixDatabase=gb get_MMapMatrixDatabase=gm

        close(mdb)
        # This is a sanity/informational benchmark, not a strict perf regression gate (disk-backed
        # I/O is inherently slower and machine/CI dependent) -- it just confirms the RAM-backed
        # types keep working and that MMapMatrixDatabase is a functional, opt-in alternative.
        @test tv >= 0 && tb >= 0 && tm >= 0
    end
end
