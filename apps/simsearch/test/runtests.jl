using Test
using JLD2, HDF5

include(joinpath(@__DIR__, "..", "src", "SimSearch.jl"))
using .SimSearch

@testset "simsearch CLI" begin
    mktempdir() do dir
        rng_data = Float32.(reshape(1:8*500, 8, 500) .% 97 ./ 97 .+ 0.001f0 .* rand(Float32, 8, 500))
        rng_queries = Float32.(rand(Float32, 8, 20))

        datafile = joinpath(dir, "data.jld2")
        JLD2.jldsave(datafile; X=rng_data)
        queryfile = joinpath(dir, "queries.jld2")
        JLD2.jldsave(queryfile; Q=rng_queries)

        dataset_spec = "$datafile:X"
        queries_spec = "$queryfile:Q"

        idx_files = Dict{String,String}()
        @testset "build" begin
            for type in ("ExhaustiveSearch", "ParallelExhaustiveSearch", "SearchGraph")
                savepath = joinpath(dir, "$(type).jld2")
                extra = type == "SearchGraph" ? ["--minrecall", "0.9"] : String[]
                SimSearch.cmd_build([
                    "--type", type, "--dataset", dataset_spec,
                    "--distance", "Dist.SqL2", "--save", savepath, extra...])
                @test isfile(savepath)
                idx_files[type] = savepath
            end
        end

        results_files = Dict{String,String}()
        @testset "search" begin
            for (type, ext) in (("ExhaustiveSearch", ".jld2"), ("SearchGraph", ".h5"))
                respath = joinpath(dir, "res_$(type)$(ext)")
                SimSearch.cmd_search([
                    idx_files[type], "--queries", queries_spec, "--dataset", dataset_spec,
                    "--results", respath, "-k", "5"])
                @test isfile(respath)
                results_files[type] = respath
            end
        end

        @testset "evaluate" begin
            htmlpath = joinpath(dir, "eval.html")
            outpath = joinpath(dir, "eval.txt")
            SimSearch.cmd_evaluate([
                "--gold", results_files["ExhaustiveSearch"],
                "--results", results_files["SearchGraph"],
                "-k", "5", "--html", htmlpath, "--out", outpath])
            @test isfile(htmlpath)
            @test isfile(outpath)
            html = read(htmlpath, String)
            @test occursin("<html", html)
            @test occursin("<svg", html)

            goldI, _ = SimSearch.load_results_spec(results_files["ExhaustiveSearch"])
            resI, _ = SimSearch.load_results_spec(results_files["SearchGraph"])
            recall = SimSearch.macrorecall(goldI, resI, 5)
            @test recall > 0.3
        end

        @testset "analyze" begin
            htmlpath = joinpath(dir, "analyze.html")
            outpath = joinpath(dir, "analyze.txt")
            SimSearch.cmd_analyze([
                idx_files["SearchGraph"], "--dataset", dataset_spec,
                "--html", htmlpath, "--out", outpath])
            @test isfile(htmlpath)
            @test isfile(outpath)
            txt = read(outpath, String)
            @test occursin("SearchGraph", txt)
            @test occursin("degree", txt)

            htmlpath2 = joinpath(dir, "analyze_exhaustive.html")
            SimSearch.cmd_analyze([
                idx_files["ExhaustiveSearch"], "--dataset", dataset_spec, "--html", htmlpath2])
            @test isfile(htmlpath2)
        end

        @testset "dataset-io roundtrip" begin
            h5file = joinpath(dir, "data.h5")
            HDF5.h5open(h5file, "w") do f
                f["X"] = rng_data
            end
            M = SimSearch.read_matrix(h5file, "X")
            @test M == rng_data
            path, key = SimSearch.split_pathkey("$h5file:X")
            @test path == h5file
            @test key == "X"
        end
    end
end
