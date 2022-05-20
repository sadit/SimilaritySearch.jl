import numpy as np
from pynndescent import NNDescent
import time
import json
import sys
import os

import pickle
from tensorflow.keras.datasets import mnist


db_name = 'mnist'
neigh = 50
div_prob = 1.0
pruning_degree_mult = 1.5
extra = "_OMP64"
if len(sys.argv) == 6:
    extra = sys.argv[5]

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
        


(data, _), (_, _) = mnist.load_data()
data = np.reshape(data, (60000, 784))
db_base = np.array(data, dtype=np.float32)
metric = 'l2'
metric_name = "L2"

    
n, d = db_base.shape
print("DB size:", n, d)


print("Building index")
start = time.time()
index = NNDescent (db_base, n_neighbors=neigh, diversify_prob=div_prob, pruning_degree_multiplier=pruning_degree_mult, metric=metric, n_jobs=64)
index.prepare()        
t_build = time.time() - start


print("Searching ...")
start = time.time()
I, D = index.query(db_base, k=k)
t_elapsed = time.time() - start


json_name = f"results.index.NNdescent{metric_name}_allknn_.neigh={neigh}.div_prob={div_prob}.prun={pruning_degree_mult}{extra}"
print("Saving index.")
with open(f"saves/{db_name}/{json_name}.pickle", "wb") as f:
    pickle.dump(index, f)
index_size = os.path.getsize(f"saves/{db_name}/{json_name}.pickle") 


txt = create_json(D, I, t_elapsed, t_build, index_size)


with open("results/" + db_name + "/" + json_name + ".json", 'w') as outfile:
    json.dump(txt, outfile)

print("The end.")
