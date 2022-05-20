using SimilaritySearch, JSON

function _summary(R, recall, filename)
    R["recall"] = recall
    R["filename"] = filename
    R["searchall"] = length(R["results"]) * R["searchtime"]
    delete!(R, "results")
    println(json(R))
end

function summary(Rflat, filename)
    R = JSON.parse(read(filename, String))
    recall = macrorecall(Rflat, [first.(r) for r in R["results"]])
    _summary(R, recall, filename)
end

let
    filename = "results.index.FlatL2_allknn_OMP64.json"
    flat = JSON.parse(read(filename, String))
    Rflat = [first.(r) for r in flat["results"]]
    _summary(flat, 1.0, filename)

    summary(Rflat, "results.index.scannL2_allknn_.leaves=0.leaves_to_search=0_OMP64.json")
    summary(Rflat, "results.index.hnswL2_allknn_.M=32.efS=32_efC40_OMP64.json")
    summary(Rflat, "results.index.NNdescentL2_allknn_.neigh=30.div_prob=1.0.prun=1.5_OMP64.json")
end
