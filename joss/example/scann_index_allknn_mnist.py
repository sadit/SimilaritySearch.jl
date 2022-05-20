import numpy as np
import scann
import time
import json
import sys
import os
from tensorflow.keras.datasets import mnist


db_name = 'mnist'
leaves = 0
leaves2search = 0
extra = "_OMP64"

k = 33
def create_json(Dist, ID, time_search, time_build, size):
    txt = {}
    txt["searchall"] = time_search
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
metric = "squared_l2"
metric_name = "L2"
    
    
    
n, d = db_base.shape
print("DB size:", n, d)


print("Building index")
start = time.time()
#index = scann.scann_ops_pybind.builder(db_base, 10, metric).tree(num_leaves=leaves, num_leaves_to_search=leaves2search, training_sample_size=n).score_ah(
#        2, anisotropic_quantization_threshold=0.2).reorder(k*10).build() # build the index
index = scann.scann_ops_pybind.builder(db_base, 10, metric).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(k*10).build() # build the index

t_build = time.time() - start


print("Searching ...")
start = time.time()
I, D = index.search_batched_parallel(db_base, final_num_neighbors=k)
t_elapsed = time.time() - start


json_name = f"results.index.scann{metric_name}_allknn_.leaves={leaves}.leaves_to_search={leaves2search}{extra}"
print("Saving index.")
os.makedirs(f'saves/{db_name}/{json_name}.index/', exist_ok=True)
index.serialize(f"saves/{db_name}/{json_name}.index/")

#index_size = sum([os.path.getsize(f) for f in os.listdir(f'saves/{db_name}/{json_name}.index/') if os.path.isfile(f)])
index_size = sum([os.path.getsize(f'saves/{db_name}/{json_name}.index/{f}') for f in os.listdir(f'saves/{db_name}/{json_name}.index/') ])
print("size:", index_size)


txt = create_json(D, I, t_elapsed, t_build, index_size)


with open("results/" + db_name + "/" + json_name + ".json", 'w') as outfile:
    json.dump(txt, outfile)

print("The end.")
