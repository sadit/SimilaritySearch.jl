import numpy as np
import faiss 
import time
import json
import sys
import os
from tensorflow.keras.datasets import mnist


db_name = 'mnist'
M = 32
efSearch = 32
extra = "_OMP64"

k = 33

def create_json(Dist, ID, time_search, time_build, size):
    txt = {}
    txt['searchall'] = time_search
    txt["searchtime"] = time_search / len(Dist)
    txt["buildtime"] = time_build
    res = []
    actual_id = 0
    for D, I  in zip(Dist, ID):
        res_q = []
        for d, i in zip(D, I):
            if (i==actual_id):
                continue
            res_q.append([int(i), float(d)])
        res.append(res_q)
        actual_id += 1
    txt["results"] = res
    txt["filesize"] = size
    return txt



metric = faiss.METRIC_L2
metric_name = "L2"

(data, _), (_, _) = mnist.load_data()
data = np.reshape(data, (60000, 784))
db_base = np.array(data, dtype=np.float32)

    
n, d = db_base.shape
print("DB size:", n, d)

print("Building index")
start = time.time()
index = faiss.IndexHNSWFlat(d, M, metric)   # build the index
index.hnsw.efConstruction = 40
index.add(db_base)
t_build = time.time() - start


print("Searching ...")
start = time.time()
index.hnsw.efSearch = efSearch
D, I = index.search(db_base, k)
t_elapsed = time.time() - start


json_name = "results.index.hnsw" + metric_name + "_allknn_.M=" + str(M) + ".efS=" + str(efSearch) + "_efC40" +extra

print("Saving index.")
faiss.write_index(index, "saves/" + db_name + "/" + json_name + ".index")
index_size = os.path.getsize("saves/" + db_name + "/" + json_name + ".index")


txt = create_json(D, I, t_elapsed, t_build, index_size)


with open("results/" + db_name + "/" + json_name + ".json", 'w') as outfile:
    json.dump(txt, outfile)

print("The end.")
