import os
import json

fname_list = ['ploy1_270523.json','ploy2_270523.json']
root_path = 'dataset'

fname_list = [os.path.join(root_path, f) for f in fname_list]

data_list = []

for fname in fname_list:
    with open(fname, "r") as f:
        data = json.load(f)
        data_list += data

target_fname = 'dataset/ploy_270523.json'
with open(target_fname, "w") as f:
    json.dump(data_list, f, indent=4)