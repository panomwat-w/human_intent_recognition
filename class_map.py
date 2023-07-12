import os
import json

fname_list = ['ploy2_120723.json']
root_path = 'dataset'

fname_list = [os.path.join(root_path, f) for f in fname_list]

data_list = []

for fname in fname_list:
    with open(fname, "r") as f:
        data = json.load(f)
        data_list += data

for data in data_list:
    if data['label'] == 0:
        data['label'] = 3
    elif data['label'] == 1:
        data['label'] = 4

target_fname = 'dataset/ploy2_120723.json'
with open(target_fname, "w") as f:
    json.dump(data_list, f, indent=4)